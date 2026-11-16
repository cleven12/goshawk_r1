"""Causal Transformer backbone for Goshawk R1 combining MLA, DeepSeekMoE, and RMSNorm."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GoshawkConfig
from .attention import MultiHeadLatentAttention
from .layers import RMSNorm
from .moe import DeepSeekMoEBlock


class GoshawkBlock(nn.Module):
    """Single Transformer Decoder layer combining MLA and DeepSeekMoE."""

    def __init__(self, config: GoshawkConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = MultiHeadLatentAttention(config)

        self.moe_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.moe = DeepSeekMoEBlock(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        # Pre-LN Multi-Head Latent Attention
        normed_attn = self.attn_norm(x)
        attn_out, new_cache = self.attn(normed_attn, mask=mask, kv_cache=kv_cache)
        h = x + attn_out

        # Pre-LN DeepSeekMoE
        normed_moe = self.moe_norm(h)
        moe_out, router_logits = self.moe(normed_moe)
        out = h + moe_out

        return out, new_cache, router_logits


class GoshawkModel(nn.Module):
    """Core Goshawk causal transformer model without LM head."""

    def __init__(self, config: GoshawkConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [GoshawkBlock(config, i) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Optional[Tuple[torch.Tensor, torch.Tensor]]], List[Optional[torch.Tensor]]]:
        batch_size, seq_len = input_ids.shape
        h = self.embed_tokens(input_ids)

        if mask is None and seq_len > 1:
            # Standard causal lower-triangular mask
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=h.device),
                diagonal=1,
            )
            mask = causal_mask.unsqueeze(0).unsqueeze(0)

        new_caches = []
        all_router_logits = []

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            h, new_c, r_logits = layer(h, mask=mask, kv_cache=layer_cache)
            new_caches.append(new_c)
            all_router_logits.append(r_logits)

        out = self.norm(h)
        return out, new_caches, all_router_logits


class GoshawkForCausalLM(nn.Module):
    """Complete Goshawk R1 model with LM prediction head and generation capability."""

    def __init__(self, config: GoshawkConfig):
        super().__init__()
        self.config = config
        self.model = GoshawkModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights between embedding and output projection
        self.lm_head.weight = self.model.embed_tokens.weight

        # Optional DeepSeek-style Multi-Token Prediction heads
        self.mtp = None
        if config.mtp_depth > 0:
            from .mtp import MultiTokenPredictor

            self.mtp = MultiTokenPredictor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        hidden_states, new_caches, all_router_logits = self.model(
            input_ids, mask=mask, kv_caches=kv_caches
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return logits, loss, new_caches

    def forward_with_mtp(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        mtp_loss_weight: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, total_loss, mtp_loss)."""
        hidden_states, _, _ = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        main_loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        mtp_loss = torch.tensor(0.0, device=input_ids.device)
        if self.mtp is not None:
            mtp_loss, _ = self.mtp.forward_train(
                backbone_hidden=hidden_states,
                input_ids=input_ids,
                embed_tokens=self.model.embed_tokens,
                lm_head=self.lm_head,
                labels=labels,
                mtp_loss_weight=mtp_loss_weight,
            )

        return logits, main_loss + mtp_loss, mtp_loss
