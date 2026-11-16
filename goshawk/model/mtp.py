"""DeepSeek-style Multi-Token Prediction (MTP) heads for Goshawk R1."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GoshawkConfig
from .layers import RMSNorm
from .transformer import GoshawkBlock


class MTPModule(nn.Module):
    """
    Single Multi-Token Prediction (MTP) depth module.

    Combines the previous hidden state representation with the next-token embedding,
    projects back to `d_model`, and passes through a transformer block to predict
    an additional future token in parallel.
    """

    def __init__(self, config: GoshawkConfig, depth_idx: int):
        super().__init__()
        self.depth_idx = depth_idx
        self.h_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.emb_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.d_model * 2, config.d_model, bias=False)
        self.block = GoshawkBlock(config, layer_idx=config.n_layers + depth_idx)

    def forward(
        self,
        prev_hidden: torch.Tensor,
        next_token_embeds: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute hidden representation for depth k+1.
        Args:
            prev_hidden: [batch, seq_len, d_model]
            next_token_embeds: [batch, seq_len, d_model]
        """
        h_normed = self.h_norm(prev_hidden)
        e_normed = self.emb_norm(next_token_embeds)
        combined = torch.cat([h_normed, e_normed], dim=-1)
        projected = self.proj(combined)
        out, _, _ = self.block(projected, mask=mask)
        return out


class MultiTokenPredictor(nn.Module):
    """
    Manages `mtp_depth` sequential MTP modules sharing the backbone's embedding
    and LM output head.
    """

    def __init__(self, config: GoshawkConfig):
        super().__init__()
        self.config = config
        self.mtp_depth = config.mtp_depth
        self.modules_list = nn.ModuleList(
            [MTPModule(config, i) for i in range(config.mtp_depth)]
        )

    def forward_train(
        self,
        backbone_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        embed_tokens: nn.Embedding,
        lm_head: nn.Linear,
        labels: torch.Tensor,
        mtp_loss_weight: float = 0.3,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute auxiliary MTP cross-entropy losses for training.
        Returns (weighted_mtp_loss, per_depth_logits).
        """
        batch_size, seq_len = input_ids.shape
        if self.mtp_depth <= 0 or seq_len <= 2:
            return torch.tensor(0.0, device=input_ids.device), []

        total_mtp_loss = torch.tensor(0.0, device=input_ids.device)
        all_logits: List[torch.Tensor] = []
        current_h = backbone_hidden
        valid_depths = 0

        for k, mtp_mod in enumerate(self.modules_list):
            shift = k + 1
            if current_h.shape[1] <= 2:
                break

            # Each step trims 1 from the tail of current_h to pair with input_ids[:, shift:]
            h_slice = current_h[:, :-1, :]
            next_ids = input_ids[:, shift:]
            next_embeds = embed_tokens(next_ids)

            sub_len = h_slice.shape[1]
            causal_mask = torch.triu(
                torch.full((sub_len, sub_len), float("-inf"), device=h_slice.device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)

            pred_h = mtp_mod(h_slice, next_embeds, mask=causal_mask)
            logits_k = lm_head(pred_h)
            all_logits.append(logits_k)

            # Target labels are shifted by one more position
            if labels is not None and sub_len > 1:
                target_slice = labels[:, shift + 1 :]
                logits_slice = logits_k[:, :-1, :].contiguous()
                loss_k = F.cross_entropy(
                    logits_slice.view(-1, self.config.vocab_size),
                    target_slice.contiguous().view(-1),
                    ignore_index=-100,
                )
                total_mtp_loss = total_mtp_loss + loss_k
                valid_depths += 1

            # Pad pred_h back to full seq_len alignment for next depth if needed
            current_h = pred_h

        if valid_depths > 0:
            total_mtp_loss = (total_mtp_loss / valid_depths) * mtp_loss_weight

        return total_mtp_loss, all_logits
