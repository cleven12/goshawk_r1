"""Tests for binary model serialization and header inspection."""

import os
import tempfile

from goshawk.config import GoshawkConfig
from goshawk.export.gguf import GoshawkBinaryExporter
from goshawk.model.transformer import GoshawkForCausalLM


def test_binary_export_and_inspect():
    config = GoshawkConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        v_head_dim=16,
        max_seq_len=32,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_dim=32,
        dense_intermediate_dim=32,
    )
    model = GoshawkForCausalLM(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "goshawk_test.gshk")
        res = GoshawkBinaryExporter.export(
            model=model,
            config=config,
            output_path=out_file,
            metadata={"test_run": True},
        )

        assert os.path.exists(out_file)
        assert res["tensors_exported"] > 0
        assert res["payload_bytes"] > 0

        info = GoshawkBinaryExporter.inspect(out_file)
        assert info["version"] == 1
        assert info["metadata"]["arch"] == "goshawk_r1"
        assert info["metadata"]["extra"]["test_run"] is True
        assert info["num_tensors"] == res["tensors_exported"]
