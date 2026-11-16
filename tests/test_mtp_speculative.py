"""Tests for Multi-Token Prediction (MTP) and Speculative Decoding."""

import torch

from goshawk.config import GoshawkConfig
from goshawk.model.generation import generate
from goshawk.model.speculative import speculative_generate
from goshawk.model.transformer import GoshawkForCausalLM


def _mini_config(d_model: int = 32, n_layers: int = 2, mtp_depth: int = 1) -> GoshawkConfig:
    return GoshawkConfig(
        vocab_size=64,
        d_model=d_model,
        n_layers=n_layers,
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
        mtp_depth=mtp_depth,
    )


def test_mtp_forward_loss():
    cfg = _mini_config(mtp_depth=2)
    model = GoshawkForCausalLM(cfg)
    assert model.mtp is not None

    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))
    labels = input_ids.clone()
    logits, total_loss, mtp_loss = model.forward_with_mtp(input_ids, labels, mtp_loss_weight=0.3)

    assert logits.shape == (2, 12, cfg.vocab_size)
    assert total_loss.item() > 0.0
    assert mtp_loss.item() > 0.0


def test_speculative_decoding_lossless_greedy():
    torch.manual_seed(42)
    cfg = _mini_config(d_model=32, n_layers=2, mtp_depth=0)
    target = GoshawkForCausalLM(cfg)
    # Use identical weights as draft to verify 100% acceptance & exact greedy match
    draft = GoshawkForCausalLM(cfg)
    draft.load_state_dict(target.state_dict())

    prompt = [3, 11, 27, 5]
    baseline = generate(target, prompt, max_new_tokens=8, temperature=0.0)
    spec_out, stats = speculative_generate(
        target, draft, prompt, max_new_tokens=8, gamma=4, temperature=0.0
    )

    assert spec_out == baseline
    assert stats.total_generated == 8
    assert stats.acceptance_rate == 1.0
    assert stats.target_forward_passes < 8


def test_speculative_decoding_mismatched_draft():
    torch.manual_seed(7)
    target = GoshawkForCausalLM(_mini_config(d_model=32, n_layers=2, mtp_depth=0))
    draft = GoshawkForCausalLM(_mini_config(d_model=32, n_layers=1, mtp_depth=0))

    prompt = [2, 9, 14]
    baseline = generate(target, prompt, max_new_tokens=6, temperature=0.0)
    spec_out, stats = speculative_generate(
        target, draft, prompt, max_new_tokens=6, gamma=3, temperature=0.0
    )

    # Even with an untrained/mismatched draft, greedy speculative decoding is lossless
    assert spec_out == baseline
    assert stats.total_generated == 6
