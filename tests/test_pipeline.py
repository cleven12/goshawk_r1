"""Smoke test for end-to-end SFT + MTP + GRPO training pipeline."""

import tempfile

from goshawk.train.pipeline import run_training_pipeline


def test_training_pipeline_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        res = run_training_pipeline(
            sft_epochs=2,
            grpo_steps=1,
            target_vocab_size=320,
            export_dir=tmpdir,
            push_to_hf=False,
        )
        assert res["total_params"] > 0
        assert res["final_sft_loss"] <= res["initial_sft_loss"] + 1.0
        assert res["export_dir"] == tmpdir
