"""Core layers for Goshawk R1: RMSNorm, SwiGLU, and Rotary Position Embeddings."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) with learnable scale."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit feedforward block."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_up = nn.Linear(in_features, hidden_features, bias=False)
        self.w_down = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with YaRN extended context scaling."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        yarn_scaling: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.yarn_scaling = yarn_scaling

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )
        if yarn_scaling != 1.0:
            inv_freq = inv_freq / yarn_scaling

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_len = offset + seq_len
        if total_len > self.cos_cached.shape[0]:
            self._build_cache(total_len)
        return self.cos_cached[offset:total_len], self.sin_cached[offset:total_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to query and key states."""
    cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed
