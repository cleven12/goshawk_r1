"""Training and GRPO reinforcement learning modules for Goshawk R1."""

from .dataset import GoshawkReasoningDataset, ReasoningSample, collate_reasoning_batch
from .grpo import (
    GRPOTrainer,
    GRPOTrainingConfig,
    compute_combined_reward,
    reward_forensic_math,
    reward_format,
    reward_legal_statute,
    reward_security_threat,
)
from .pretrain import GoshawkTrainer, PretrainConfig

__all__ = [
    "GoshawkReasoningDataset",
    "ReasoningSample",
    "collate_reasoning_batch",
    "GRPOTrainer",
    "GRPOTrainingConfig",
    "compute_combined_reward",
    "reward_format",
    "reward_forensic_math",
    "reward_legal_statute",
    "reward_security_threat",
    "GoshawkTrainer",
    "PretrainConfig",
]
