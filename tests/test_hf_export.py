"""Tests for Hugging Face Hub export bundle generator."""

import json
import os
import tempfile

from goshawk.config import GoshawkConfig
from goshawk.export.hf_hub import export_hf_bundle
from goshawk.model.transformer import GoshawkForCausalLM
from goshawk.tokenizer.bpe import GoshawkTokenizer


def test_export_hf_bundle():
    cfg = GoshawkConfig(
        vocab_size=512,
        d_model=32,
        n_layers=1,
        n_heads=2,
        v_head_dim=16,
        max_seq_len=64,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_dim=32,
        dense_intermediate_dim=32,
        mtp_depth=0,
    )
    model = GoshawkForCausalLM(cfg)
    tok = GoshawkTokenizer.default()

    with tempfile.TemporaryDirectory() as tmpdir:
        res = export_hf_bundle(model, cfg, tmpdir, tokenizer=tok, model_name="cleven12/goshawk-r1-nano")
        expected = {
            "README.md",
            "config.json",
            "merges.json",
            "model.gshk",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "vocab.json",
        }
        assert expected.issubset(set(res["files"]))

        with open(os.path.join(tmpdir, "config.json"), "r", encoding="utf-8") as f:
            hf_cfg = json.load(f)
        assert hf_cfg["model_type"] == "goshawk_r1"
        assert hf_cfg["kv_lora_rank"] == 8
