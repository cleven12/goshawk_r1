"""Autoregressive generation routines with MLA KV caching, prefix caching, and top-p sampling."""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .transformer import GoshawkForCausalLM

KVCacheList = List[Tuple[torch.Tensor, torch.Tensor]]


class LatentPrefixCache:
    """
    LRU prefix cache storing compressed MLA latent KV states (c_kv, k_rope).
    Enables near-instant Time-To-First-Token (TTFT) for recurring system prompts,
    Tanzanian legal statutes, and CCTV camera metadata.
    """

    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._cache: OrderedDict[Tuple[int, ...], KVCacheList] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _clone_cache(self, kv_caches: KVCacheList) -> KVCacheList:
        return [(c_kv.clone(), k_rope.clone()) for c_kv, k_rope in kv_caches]

    def put(self, prefix_tokens: List[int], kv_caches: KVCacheList) -> None:
        """Store compressed latent KV states for a token prefix."""
        key = tuple(prefix_tokens)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = self._clone_cache(kv_caches)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def lookup_longest(self, tokens: List[int]) -> Tuple[int, Optional[KVCacheList]]:
        """
        Find the longest cached prefix matching `tokens[:-1]`.
        Leaves at least 1 token to run through the model to emit next-token logits.
        """
        max_match_len = len(tokens) - 1
        if max_match_len <= 0:
            self.misses += 1
            return 0, None

        best_len = 0
        best_key: Optional[Tuple[int, ...]] = None
        for key in self._cache:
            klen = len(key)
            if klen <= max_match_len and klen > best_len and tuple(tokens[:klen]) == key:
                best_len = klen
                best_key = key

        if best_key is not None:
            self._cache.move_to_end(best_key)
            self.hits += 1
            return best_len, self._clone_cache(self._cache[best_key])

        self.misses += 1
        return 0, None

    @property
    def stats(self) -> Dict[str, int]:
        return {"entries": len(self._cache), "hits": self.hits, "misses": self.misses}


def sample_top_p(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 0.7) -> torch.Tensor:
    """Sample next token from logits using nucleus (top-p) sampling."""
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / max(temperature, 1e-5)
    probs = F.softmax(scaled_logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    scaled_logits[indices_to_remove] = float("-inf")
    filtered_probs = F.softmax(scaled_logits, dim=-1)

    return torch.multinomial(filtered_probs, num_samples=1)


@torch.no_grad()
def generate(
    model: GoshawkForCausalLM,
    prompt_tokens: List[int],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop_tokens: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    prefix_cache: Optional[LatentPrefixCache] = None,
    cache_prefix_len: Optional[int] = None,
) -> List[int]:
    """Generate tokens autoregressively from a prompt using Goshawk R1."""
    dev = device or next(model.parameters()).device
    model.eval()

    stop_set = set(stop_tokens or [])
    generated: List[int] = list(prompt_tokens)

    kv_caches: Optional[KVCacheList] = None
    matched_len = 0

    if prefix_cache is not None and len(prompt_tokens) > 1:
        matched_len, kv_caches = prefix_cache.lookup_longest(prompt_tokens)

    # Prefill uncached suffix (if matched_len > 0, build causal mask for suffix against full key len)
    uncached = prompt_tokens[matched_len:]
    if kv_caches is not None and len(uncached) > 1:
        q_len = len(uncached)
        k_len = matched_len + q_len
        mask = torch.zeros((1, 1, q_len, k_len), device=dev)
        mask[:, :, :, matched_len:] = torch.triu(
            torch.full((q_len, q_len), float("-inf"), device=dev),
            diagonal=1,
        )
        current_ids = torch.tensor([uncached], dtype=torch.long, device=dev)
        logits, _, kv_caches = model(current_ids, mask=mask, kv_caches=kv_caches)
    else:
        current_ids = torch.tensor([uncached], dtype=torch.long, device=dev)
        logits, _, kv_caches = model(current_ids, kv_caches=kv_caches)

    # Optionally store prompt prefix in cache for subsequent queries
    if prefix_cache is not None and cache_prefix_len is not None and cache_prefix_len > 0:
        if cache_prefix_len <= len(prompt_tokens) and kv_caches is not None:
            prefix_kv = [(c[:, :cache_prefix_len, :], kr[:, :, :cache_prefix_len, :]) for c, kr in kv_caches]
            prefix_cache.put(prompt_tokens[:cache_prefix_len], prefix_kv)

    for step in range(max_new_tokens):
        if step > 0:
            logits, _, kv_caches = model(current_ids, kv_caches=kv_caches)

        next_token_logits = logits[:, -1, :]
        next_token_tensor = sample_top_p(next_token_logits, top_p=top_p, temperature=temperature)
        next_token = int(next_token_tensor.item())

        generated.append(next_token)
        if next_token in stop_set:
            break

        current_ids = next_token_tensor

    return generated
