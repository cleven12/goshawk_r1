"""Speculative decoding engine for Goshawk R1 (Nano draft -> Edge target verification)."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .generation import sample_top_p
from .transformer import GoshawkForCausalLM

KVCacheList = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class SpeculativeStats:
    """Telemetry for speculative decoding efficiency."""

    total_generated: int
    drafted_tokens: int
    accepted_tokens: int
    target_forward_passes: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(self.drafted_tokens, 1)

    @property
    def tokens_per_target_pass(self) -> float:
        return self.total_generated / max(self.target_forward_passes, 1)


def _truncate_kv_cache(kv_caches: Optional[KVCacheList], keep_len: int) -> Optional[KVCacheList]:
    """Slice MLA latent KV caches back to `keep_len` positions after draft rejection."""
    if kv_caches is None:
        return None
    truncated: KVCacheList = []
    for c_kv, k_rope in kv_caches:
        truncated.append((c_kv[:, :keep_len, :], k_rope[:, :, :keep_len, :]))
    return truncated


@torch.no_grad()
def speculative_generate(
    target_model: GoshawkForCausalLM,
    draft_model: GoshawkForCausalLM,
    prompt_tokens: List[int],
    max_new_tokens: int = 64,
    gamma: int = 4,
    temperature: float = 0.0,
    top_p: float = 0.9,
    stop_tokens: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[int], SpeculativeStats]:
    """
    Generate tokens using speculative decoding:
    1. `draft_model` (e.g. Goshawk-Nano) autoregressively proposes `gamma` tokens.
    2. `target_model` (e.g. Goshawk-Edge) verifies all `gamma` tokens in one parallel pass.
    3. Lossless rejection sampling accepts matching prefix + 1 bonus token from target.
    """
    dev = device or next(target_model.parameters()).device
    target_model.eval()
    draft_model.eval()

    stop_set = set(stop_tokens or [])
    generated: List[int] = list(prompt_tokens)
    if max_new_tokens <= 0:
        return generated, SpeculativeStats(0, 0, 0, 0)

    # Prefill both models on prompt[:-1], keeping prompt[-1] as the active boundary token
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=dev)
    target_cache: Optional[KVCacheList] = None
    draft_cache: Optional[KVCacheList] = None

    if len(prompt_tokens) > 1:
        prefix = prompt_tensor[:, :-1]
        _, _, target_cache = target_model(prefix, kv_caches=None)
        _, _, draft_cache = draft_model(prefix, kv_caches=None)

    last_token = prompt_tensor[:, -1:]  # [1, 1]

    drafted_total = 0
    accepted_total = 0
    target_passes = 0
    tokens_produced = 0

    while tokens_produced < max_new_tokens:
        step_gamma = min(gamma, max_new_tokens - tokens_produced)
        base_len = len(generated) - 1  # Tokens already in KV cache

        # 1. Draft `step_gamma` candidate tokens
        draft_tokens: List[int] = []
        draft_probs: List[torch.Tensor] = []
        cur_draft_in = last_token
        cur_draft_cache = draft_cache

        for _ in range(step_gamma):
            d_logits, _, cur_draft_cache = draft_model(cur_draft_in, kv_caches=cur_draft_cache)
            next_d_logits = d_logits[:, -1, :]
            if temperature <= 0.0:
                d_tok_tensor = torch.argmax(next_d_logits, dim=-1, keepdim=True)
                d_prob = F.softmax(next_d_logits, dim=-1)
            else:
                d_prob = F.softmax(next_d_logits / max(temperature, 1e-5), dim=-1)
                d_tok_tensor = sample_top_p(next_d_logits, top_p=top_p, temperature=temperature)

            draft_tokens.append(int(d_tok_tensor.item()))
            draft_probs.append(d_prob.squeeze(0))
            cur_draft_in = d_tok_tensor

        drafted_total += len(draft_tokens)

        # 2. Verify all draft tokens in ONE parallel target forward pass
        verify_seq = torch.cat(
            [last_token, torch.tensor([draft_tokens], dtype=torch.long, device=dev)],
            dim=1,
        )  # [1, step_gamma + 1]

        # Causal mask for verify_seq when cache is present (query_len = g+1, key_len = base_len + g + 1)
        q_len = verify_seq.shape[1]
        k_len = base_len + q_len
        verify_mask = torch.zeros((1, 1, q_len, k_len), device=dev)
        future_mask = torch.triu(
            torch.full((q_len, q_len), float("-inf"), device=dev),
            diagonal=1,
        )
        verify_mask[:, :, :, base_len:] = future_mask

        t_logits, _, new_target_cache = target_model(
            verify_seq, mask=verify_mask, kv_caches=target_cache
        )
        target_passes += 1

        # 3. Speculative acceptance check
        accept_len = 0
        resampled_token: Optional[int] = None

        for i, d_tok in enumerate(draft_tokens):
            t_logits_i = t_logits[:, i, :]
            if temperature <= 0.0:
                t_best = int(torch.argmax(t_logits_i, dim=-1).item())
                if t_best == d_tok:
                    accept_len += 1
                    accepted_total += 1
                else:
                    resampled_token = t_best
                    break
            else:
                t_prob = F.softmax(t_logits_i / max(temperature, 1e-5), dim=-1).squeeze(0)
                p_t = float(t_prob[d_tok].item())
                p_d = max(float(draft_probs[i][d_tok].item()), 1e-9)
                accept_prob = min(1.0, p_t / p_d)
                if torch.rand(1, device=dev).item() < accept_prob:
                    accept_len += 1
                    accepted_total += 1
                else:
                    diff = torch.clamp(t_prob - draft_probs[i], min=0.0)
                    if diff.sum() > 0:
                        diff = diff / diff.sum()
                        resampled_token = int(torch.multinomial(diff, num_samples=1).item())
                    else:
                        resampled_token = int(torch.argmax(t_prob).item())
                    break

        # Append accepted draft tokens
        hit_stop = False
        for tok in draft_tokens[:accept_len]:
            generated.append(tok)
            tokens_produced += 1
            if tok in stop_set or tokens_produced >= max_new_tokens:
                hit_stop = True
                break

        if hit_stop:
            break

        # Append bonus token (either resampled on rejection or sampled from t_logits[accept_len])
        if resampled_token is not None:
            bonus_tok = resampled_token
        else:
            bonus_logits = t_logits[:, accept_len, :]
            bonus_tok = int(
                sample_top_p(bonus_logits, top_p=top_p, temperature=temperature).item()
            )

        generated.append(bonus_tok)
        tokens_produced += 1
        if bonus_tok in stop_set or tokens_produced >= max_new_tokens:
            break

        # 4. Sync KV caches to exact accepted length (base_len + 1 + accept_len)
        keep_len = base_len + 1 + accept_len
        target_cache = _truncate_kv_cache(new_target_cache, keep_len)
        draft_cache = _truncate_kv_cache(cur_draft_cache, keep_len)
        last_token = torch.tensor([[bonus_tok]], dtype=torch.long, device=dev)

    stats = SpeculativeStats(
        total_generated=tokens_produced,
        drafted_tokens=drafted_total,
        accepted_tokens=accepted_total,
        target_forward_passes=target_passes,
    )
    return generated, stats
