"""Tests for dataset loading, batch collation, and pretraining engine."""

import pytest

from goshawk.config import GoshawkConfig
from goshawk.model.transformer import GoshawkForCausalLM
from goshawk.tokenizer.bpe import GoshawkTokenizer
from goshawk.train.dataset import (
    SYNTHETIC_REASONING_SAMPLES,
    GoshawkReasoningDataset,
    collate_reasoning_batch,
)
from goshawk.train.pretrain import GoshawkTrainer, PretrainConfig, get_cosine_lr


def test_dataset_and_collation():
    tokenizer = GoshawkTokenizer.default()
    dataset = GoshawkReasoningDataset(tokenizer=tokenizer, max_seq_len=256)
    assert len(dataset) == len(SYNTHETIC_REASONING_SAMPLES)

    item = dataset[0]
    assert "input_ids" in item
    assert "labels" in item
    assert item["input_ids"].shape == item["labels"].shape

    batch = [dataset[0], dataset[1]]
    collated = collate_reasoning_batch(batch)
    assert collated["input_ids"].shape[0] == 2
    assert collated["attention_mask"].shape[0] == 2


def test_cosine_lr_schedule():
    lr = get_cosine_lr(step=0, total_steps=100, warmup_steps=10, lr=1e-3, min_lr=1e-5)
    assert lr < 1e-3  # During warmup

    lr_peak = get_cosine_lr(step=10, total_steps=100, warmup_steps=10, lr=1e-3, min_lr=1e-5)
    assert pytest.approx(lr_peak, rel=1e-2) == 1e-3

    lr_end = get_cosine_lr(step=110, total_steps=100, warmup_steps=10, lr=1e-3, min_lr=1e-5)
    assert lr_end == 1e-5


def test_trainer_single_epoch():
    config = GoshawkConfig(
        vocab_size=512,
        d_model=64,
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

    # Synthetic mini-samples with small token IDs
    mini_samples = [
        {
            "prompt": "Test prompt 1",
            "thinking": "<think>Step 1</think>",
            "response": "Response 1",
            "task_type": "forensics",
        },
        {
            "prompt": "Test prompt 2",
            "thinking": "<think>Step 2</think>",
            "response": "Response 2",
            "task_type": "security_event",
        },
    ]

    dataset = GoshawkReasoningDataset(samples=mini_samples, max_seq_len=64)
    trainer = GoshawkTrainer(
        model=model,
        train_dataset=dataset,
        config=PretrainConfig(batch_size=2, max_epochs=1, gradient_accumulation_steps=1),
    )

    metrics = trainer.train_epoch(epoch=1)
    assert "train_loss" in metrics
    assert metrics["train_loss"] > 0.0
