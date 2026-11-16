"""Training and GRPO reinforcement learning modules for Goshawk R1."""

from .corpus import TRAINING_CORPUS
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
from .pipeline import build_edge_trained_config, run_training_pipeline
from .pretrain import GoshawkTrainer, PretrainConfig

__all__ = [
    "TRAINING_CORPUS",
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
    "build_edge_trained_config",
    "run_training_pipeline",
]
