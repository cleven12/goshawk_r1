"""Autoregressive generation routines with KV caching and sampling options for Goshawk R1."""

from typing import List, Optional

import torch
import torch.nn.functional as F

from .transformer import GoshawkForCausalLM


def sample_top_p(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 0.7) -> torch.Tensor:
    """Sample next token from logits using nucleus (top-p) sampling."""
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / max(temperature, 1e-5)
    probs = F.softmax(scaled_logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift indices right to keep first token above threshold
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
) -> List[int]:
    """Generate tokens autoregressively from a prompt using Goshawk R1."""
    dev = device or next(model.parameters()).device
    model.eval()

    current_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=dev)
    stop_set = set(stop_tokens or [])
    generated: List[int] = list(prompt_tokens)

    kv_caches = None

    for _ in range(max_new_tokens):
        # In prefill or continuation step
        logits, _, kv_caches = model(current_ids, kv_caches=kv_caches)
        next_token_logits = logits[:, -1, :]

        next_token_tensor = sample_top_p(next_token_logits, top_p=top_p, temperature=temperature)
        next_token = int(next_token_tensor.item())

        generated.append(next_token)
        if next_token in stop_set:
            break

        current_ids = next_token_tensor

    return generated
