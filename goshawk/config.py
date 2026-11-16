"""Model configuration parameters for Goshawk R1 (Nano, Micro, and Edge profiles)."""

from dataclasses import dataclass


@dataclass
class GoshawkConfig:
    """Hyperparameters for Goshawk R1 model architecture."""

    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    v_head_dim: int = 64
    max_seq_len: int = 4096

    # Multi-Head Latent Attention (MLA) parameters
    q_lora_rank: int = 256
    kv_lora_rank: int = 128
    qk_rope_head_dim: int = 32

    # DeepSeekMoE parameters
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    moe_intermediate_dim: int = 1536
    dense_intermediate_dim: int = 2816

    # RoPE context parameters
    rope_theta: float = 10000.0
    yarn_scaling_factor: float = 1.0

    # Normalization and dropout
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0

    # Multi-Token Prediction (MTP) depth (DeepSeek-V3 style)
    mtp_depth: int = 1

    @classmethod
    def nano(cls) -> "GoshawkConfig":
        """
        Ultra-compact profile (~180M parameters, ~45M active per token).
        Optimized for Raspberry Pi 5, edge CCTV DVRs, and real-time security alerts.
        """
        return cls(
            vocab_size=16384,
            d_model=512,
            n_layers=10,
            n_heads=8,
            v_head_dim=48,
            max_seq_len=2048,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_rope_head_dim=24,
            n_routed_experts=6,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_dim=768,
            dense_intermediate_dim=1376,
        )

    @classmethod
    def micro(cls) -> "GoshawkConfig":
        """
        Medium-compact profile (~850M parameters, ~210M active per token).
        Optimized for standard consumer laptops (Intel i5, Ryzen 5, 4GB RAM) without GPU.
        Ideal for mobile money and bank forensic audits.
        """
        return cls(
            vocab_size=32000,
            d_model=1024,
            n_layers=16,
            n_heads=16,
            v_head_dim=64,
            max_seq_len=4096,
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_rope_head_dim=32,
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_dim=1536,
            dense_intermediate_dim=2816,
        )

    @classmethod
    def edge(cls) -> "GoshawkConfig":
        """
        Full edge workstation profile (~2.8B parameters, ~650M active per token).
        Optimized for law firm servers and multi-camera municipal setups.
        """
        return cls(
            vocab_size=32000,
            d_model=2048,
            n_layers=24,
            n_heads=16,
            v_head_dim=128,
            max_seq_len=8192,
            q_lora_rank=512,
            kv_lora_rank=256,
            qk_rope_head_dim=64,
            n_routed_experts=16,
            n_shared_experts=2,
            num_experts_per_tok=3,
            moe_intermediate_dim=3072,
            dense_intermediate_dim=5504,
        )
