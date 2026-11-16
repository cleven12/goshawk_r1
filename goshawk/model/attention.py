"""DeepSeek-style Multi-Head Latent Attention (MLA) with low-rank Key/Value compression."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GoshawkConfig
from .layers import RotaryEmbedding, apply_rotary_pos_emb


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) module.

    Compacts Key-Value states into a compressed latent vector (c_kv) of dimension `kv_lora_rank`,
    reducing inference memory by ~80% compared to standard Multi-Head Attention (MHA).
    Decouples rotary position embeddings (RoPE) into a separate dimension for position awareness.
    """

    def __init__(self, config: GoshawkConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.v_head_dim = config.v_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        # Query low-rank down/up projection
        self.wq_down = nn.Linear(self.d_model, self.q_lora_rank, bias=False)
        self.q_norm = nn.LayerNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_up_nope = nn.Linear(
            self.q_lora_rank, self.n_heads * self.v_head_dim, bias=False
        )
        self.wq_up_rope = nn.Linear(
            self.q_lora_rank, self.n_heads * self.qk_rope_head_dim, bias=False
        )

        # Key-Value low-rank down/up projection
        self.wkv_down = nn.Linear(self.d_model, self.kv_lora_rank, bias=False)
        self.kv_norm = nn.LayerNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.wkv_up_nope = nn.Linear(
            self.kv_lora_rank, self.n_heads * self.v_head_dim, bias=False
        )
        self.wk_rope = nn.Linear(self.d_model, self.qk_rope_head_dim, bias=False)

        # Output projection
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.d_model, bias=False)

        # RoPE embedding
        self.rotary_emb = RotaryEmbedding(
            dim=self.qk_rope_head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_theta,
            yarn_scaling=config.yarn_scaling_factor,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        offset = kv_cache[0].shape[1] if kv_cache is not None else 0

        # 1. Project Query through Low-Rank Bottleneck
        q_latent = self.q_norm(self.wq_down(x))
        q_nope = self.wq_up_nope(q_latent).view(
            batch_size, seq_len, self.n_heads, self.v_head_dim
        ).transpose(1, 2)
        q_rope = self.wq_up_rope(q_latent).view(
            batch_size, seq_len, self.n_heads, self.qk_rope_head_dim
        ).transpose(1, 2)

        # 2. Project Key-Value Latent state
        c_kv = self.kv_norm(self.wkv_down(x))
        kv_nope = self.wkv_up_nope(c_kv).view(
            batch_size, seq_len, self.n_heads, self.v_head_dim
        ).transpose(1, 2)

        k_rope = self.wk_rope(x).view(
            batch_size, seq_len, 1, self.qk_rope_head_dim
        ).transpose(1, 2).expand(-1, self.n_heads, -1, -1)

        # 3. Apply RoPE to Decoupled Heads at current offset
        cos, sin = self.rotary_emb(x, seq_len, offset=offset)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)

        # 4. Handle KV Cache
        if kv_cache is not None:
            prev_c_kv, prev_k_rope = kv_cache
            c_kv = torch.cat([prev_c_kv, c_kv], dim=1)
            k_rope = torch.cat([prev_k_rope, k_rope], dim=2)
            # Recompute full Key-Value nope from concatenated latent state
            kv_nope = self.wkv_up_nope(c_kv).view(
                batch_size, -1, self.n_heads, self.v_head_dim
            ).transpose(1, 2)
            new_cache = (c_kv, k_rope)
        else:
            new_cache = (c_kv, k_rope)

        # 5. Concatenate NOPE and ROPE keys
        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([kv_nope, k_rope], dim=-1)
        v_full = kv_nope

        # 6. Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.v_head_dim + self.qk_rope_head_dim)
        scores = torch.matmul(q_full, k_full.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v_full)

        # 7. Output Projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.v_head_dim
        )
        output = self.wo(attn_out)
        return output, new_cache
