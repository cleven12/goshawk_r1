"""DeepSeekMoE block with isolated shared experts, fine-grained routed experts, and top-k gating."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GoshawkConfig
from .layers import SwiGLU


class DeepSeekMoEBlock(nn.Module):
    """
    Mixture-of-Experts block following the DeepSeekMoE architecture.

    - Shared Experts (always active): Maintains baseline cross-task continuity and grammatical rules.
    - Routed Experts (fine-grained): Dynamically selected via learned routing logits.
    - Top-K Gating: Dispatches token representations to the top scoring specialists.
    """

    def __init__(self, config: GoshawkConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.moe_dim = config.moe_intermediate_dim

        # Gating network
        self.gate = nn.Linear(self.d_model, self.n_routed, bias=False)

        # Shared experts (execute unconditionally)
        self.shared_experts = nn.ModuleList(
            [SwiGLU(self.d_model, self.moe_dim) for _ in range(self.n_shared)]
        )

        # Fine-grained routed experts
        self.routed_experts = nn.ModuleList(
            [SwiGLU(self.d_model, self.moe_dim) for _ in range(self.n_routed)]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_dim = x.shape
        flat_x = x.view(-1, hidden_dim)

        # 1. Always evaluate Shared Experts
        shared_output = torch.zeros_like(flat_x)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(flat_x)

        # 2. Gate Routing Logits
        router_logits = self.gate(flat_x)
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select Top-K Experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        # Normalize weights across selected experts
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 3. Dispatch to Routed Experts
        routed_output = torch.zeros_like(flat_x)
        for k in range(self.top_k):
            indices_k = topk_indices[:, k]
            weights_k = topk_weights[:, k].unsqueeze(-1)

            for expert_idx in range(self.n_routed):
                mask = indices_k == expert_idx
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_res = self.routed_experts[expert_idx](expert_input)
                    routed_output[mask] += expert_res * weights_k[mask]

        # 4. Combine Shared and Routed Outputs
        total_output = (shared_output + routed_output).view(
            batch_size, seq_len, hidden_dim
        )
        return total_output, router_logits
