"""Hugging Face Hub export bundle generator for Goshawk R1."""

import json
import os
from typing import Any, Dict, Optional

import torch

from ..config import GoshawkConfig
from ..model.transformer import GoshawkForCausalLM
from ..tokenizer.bpe import GoshawkTokenizer
from ..tokenizer.vocab import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID
from .gguf import GoshawkBinaryExporter


def build_hf_config(config: GoshawkConfig) -> Dict[str, Any]:
    """Build a Hugging Face-compatible config.json dictionary for Goshawk R1."""
    return {
        "architectures": ["GoshawkForCausalLM"],
        "model_type": "goshawk_r1",
        "vocab_size": config.vocab_size,
        "hidden_size": config.d_model,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "v_head_dim": config.v_head_dim,
        "max_position_embeddings": config.max_seq_len,
        "q_lora_rank": config.q_lora_rank,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "n_routed_experts": config.n_routed_experts,
        "n_shared_experts": config.n_shared_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "moe_intermediate_size": config.moe_intermediate_dim,
        "intermediate_size": config.dense_intermediate_dim,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.rms_norm_eps,
        "mtp_depth": config.mtp_depth,
        "bos_token_id": BOS_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "unk_token_id": UNK_TOKEN_ID,
        "torch_dtype": "float32",
    }


def build_model_card(model_name: str, config: GoshawkConfig) -> str:
    """Generate a Hugging Face Model Card README.md with YAML metadata."""
    return f"""---
license: apache-2.0
language:
- en
- sw
tags:
- goshawk-r1
- deepseek-mla
- deepseek-moe
- reasoning
- edge-security
- digital-forensics
- tanzania
---

# {model_name}

**Goshawk R1** is an edge-optimized reasoning language model combining DeepSeek Multi-Head Latent Attention (MLA), DeepSeekMoE, and Group Relative Policy Optimization (GRPO).

## Architecture Configuration

- **Hidden Size (`d_model`)**: {config.d_model}
- **Layers (`n_layers`)**: {config.n_layers}
- **Attention Heads (`n_heads`)**: {config.n_heads}
- **MLA KV Latent Rank (`kv_lora_rank`)**: {config.kv_lora_rank}
- **Routed Experts**: {config.n_routed_experts} (Top-{config.num_experts_per_tok} active) + {config.n_shared_experts} Shared
- **Multi-Token Prediction Depth**: {config.mtp_depth}
"""


def export_hf_bundle(
    model: GoshawkForCausalLM,
    config: GoshawkConfig,
    output_dir: str,
    tokenizer: Optional[GoshawkTokenizer] = None,
    model_name: str = "cleven12/goshawk-r1-nano",
) -> Dict[str, Any]:
    """
    Export complete Hugging Face-ready directory containing:
    - config.json
    - tokenizer_config.json, vocab.json, merges.json
    - pytorch_model.bin
    - model.gshk (native edge binary)
    - README.md (HF Model Card)
    """
    os.makedirs(output_dir, exist_ok=True)
    tok = tokenizer or GoshawkTokenizer.default()

    # 1. config.json
    hf_cfg = build_hf_config(config)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_cfg, f, indent=2)

    # 2. tokenizer files
    tok.save(output_dir)
    tok_cfg = {
        "tokenizer_class": "GoshawkTokenizer",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": config.max_seq_len,
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2)

    # 3. PyTorch weights + native .gshk binary
    pt_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), pt_path)

    gshk_path = os.path.join(output_dir, "model.gshk")
    GoshawkBinaryExporter.export(model, config, gshk_path, metadata={"hf_repo": model_name})

    # 4. HF Model Card
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(build_model_card(model_name, config))

    return {
        "output_dir": output_dir,
        "files": sorted(os.listdir(output_dir)),
        "model_name": model_name,
    }
