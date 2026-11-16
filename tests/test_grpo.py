"""Tests for DeepSeek-R1 style Group Relative Policy Optimization (GRPO) logic and rewards."""

import torch

from goshawk.config import GoshawkConfig
from goshawk.model.transformer import GoshawkForCausalLM
from goshawk.train.grpo import (
    GRPOTrainer,
    GRPOTrainingConfig,
    reward_forensic_math,
    reward_format,
    reward_legal_statute,
    reward_security_threat,
)


def test_reward_format():
    valid_completion = "<think>Step 1: Check Section 40A.\nStep 2: Compare balances.</think>\nResult: Valid evidence."
    assert reward_format(valid_completion) == 1.0

    missing_tags = "Here is the direct answer without any reasoning."
    assert reward_format(missing_tags) == -1.0

    empty_think = "<think></think>Direct answer"
    assert reward_format(empty_think) < 1.0


def test_reward_forensic_math():
    prompt = "Account received 5,000,000 TZS. Sent 4,500,000 TZS to counterparty."
    good_completion = "<think>Inflow: 5,000,000 and outflow 4,500,000.</think>Confirmed."
    assert reward_forensic_math(prompt, good_completion) == 1.0

    no_math = "<think>Everything looks suspicious.</think>Alert."
    assert reward_forensic_math(prompt, no_math) == 0.0


def test_reward_legal_statute():
    completion = "Under Section 40A of the Evidence Act and Section 302 of the Penal Code."
    score = reward_legal_statute(completion)
    assert score >= 0.5


def test_reward_security_threat():
    prompt = "Subject observed entering with pry bar and crowbar."
    good_completion = "Threat level CRITICAL: Forced entry detected."
    assert reward_security_threat(prompt, good_completion) == 1.0

    bad_completion = "Threat level LOW: Normal entry."
    assert reward_security_threat(prompt, bad_completion) < 0.0


def test_grpo_advantages_standardization():
    config = GoshawkConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
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
    )
    model = GoshawkForCausalLM(config)
    trainer = GRPOTrainer(model)

    rewards = [1.0, 0.5, 0.0, -0.5]
    advantages = trainer.calculate_advantages(rewards)

    assert len(advantages) == 4
    assert torch.isclose(advantages.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(advantages.std(unbiased=False), torch.tensor(1.0), atol=1e-3)


def test_grpo_loss_computation():
    config = GoshawkConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
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
    )
    model = GoshawkForCausalLM(config)
    trainer = GRPOTrainer(model)

    log_probs_policy = torch.tensor([-1.2, -0.8, -2.1, -0.5])
    log_probs_old = torch.tensor([-1.1, -0.9, -2.0, -0.6])
    advantages = torch.tensor([1.2, 0.4, -0.8, -0.8])

    loss, metrics = trainer.compute_grpo_loss(log_probs_policy, log_probs_old, advantages)
    assert isinstance(loss, torch.Tensor)
    assert "policy_loss" in metrics
    assert "mean_ratio" in metrics


def test_grpo_train_step_end_to_end():
    config = GoshawkConfig(
        vocab_size=512,
        d_model=32,
        n_layers=1,
        n_heads=2,
        v_head_dim=16,
        max_seq_len=128,
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
    policy = GoshawkForCausalLM(config)
    ref = GoshawkForCausalLM(config)
    ref.load_state_dict(policy.state_dict())

    grpo_cfg = GRPOTrainingConfig(group_size=2, max_gen_tokens=6)
    trainer = GRPOTrainer(policy_model=policy, ref_model=ref, config=grpo_cfg)

    metrics = trainer.train_step("<forensic> Audit TZS 500,000. </forensic>")
    assert "mean_reward" in metrics
    assert "total_loss" in metrics
    assert "kl_divergence" in metrics
