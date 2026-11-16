"""End-to-end SFT + MTP + DeepSeek-R1 GRPO training and Hugging Face publishing pipeline."""

import copy
import time
from typing import Any, Dict, List, Optional

import torch

from ..config import GoshawkConfig
from ..export.hf_hub import export_hf_bundle, upload_hf_bundle
from ..model.generation import LatentPrefixCache, generate
from ..model.transformer import GoshawkForCausalLM
from ..tokenizer.bpe import GoshawkTokenizer
from ..tokenizer.vocab import EOS_TOKEN_ID
from .corpus import TRAINING_CORPUS
from .dataset import SYNTHETIC_REASONING_SAMPLES, GoshawkReasoningDataset, collate_reasoning_batch
from .grpo import GRPOTrainer, GRPOTrainingConfig


def build_edge_trained_config(vocab_size: int) -> GoshawkConfig:
    """Compact MLA + DeepSeekMoE + MTP architecture tuned for fast edge convergence."""
    return GoshawkConfig(
        vocab_size=vocab_size,
        d_model=192,
        n_layers=4,
        n_heads=4,
        v_head_dim=48,
        max_seq_len=512,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_dim=256,
        dense_intermediate_dim=384,
        mtp_depth=1,
    )


def run_training_pipeline(
    sft_epochs: int = 35,
    grpo_steps: int = 6,
    target_vocab_size: int = 512,
    export_dir: Optional[str] = None,
    push_to_hf: bool = False,
    hf_repo_id: str = "Cleven12/goshawk-r1-nano",
    env_path: str = ".env",
) -> Dict[str, Any]:
    """
    Execute full Goshawk R1 training pipeline:
    1. Train BPE tokenizer on East African forensic & CCTV security corpus.
    2. Train GoshawkForCausalLM with SFT + Multi-Token Prediction (MTP) auxiliary loss.
    3. Refine reasoning via DeepSeek-R1 Group Relative Policy Optimization (GRPO).
    4. Verify live <think>...</think> generation and optionally publish to Hugging Face.
    """
    all_samples = list(TRAINING_CORPUS) + list(SYNTHETIC_REASONING_SAMPLES)
    corpus_texts = [
        f"{s['prompt']}\n{s['thinking']}\n{s['response']}" for s in all_samples
    ]

    t0 = time.perf_counter()
    tokenizer = GoshawkTokenizer.train(corpus_texts, target_vocab_size=target_vocab_size)
    config = build_edge_trained_config(vocab_size=tokenizer.vocab_size)

    model = GoshawkForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())

    dataset = GoshawkReasoningDataset(
        samples=all_samples,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_reasoning_batch,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(sft_epochs, 1))

    # --- Stage 1: Supervised Fine-Tuning + Multi-Token Prediction (MTP) ---
    sft_losses: List[float] = []
    model.train()
    for _ in range(sft_epochs):
        epoch_loss = 0.0
        steps = 0
        for batch in loader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            optimizer.zero_grad()
            _, total_loss, _ = model.forward_with_mtp(input_ids, labels=labels, mtp_loss_weight=0.2)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(total_loss.item())
            steps += 1
        scheduler.step()
        sft_losses.append(epoch_loss / max(steps, 1))

    # --- Stage 2: DeepSeek-R1 GRPO Reinforcement Learning ---
    ref_model = copy.deepcopy(model).eval()
    grpo_cfg = GRPOTrainingConfig(
        group_size=3,
        max_gen_tokens=48,
        learning_rate=2e-5,
        temperature=0.7,
    )
    grpo = GRPOTrainer(policy_model=model, ref_model=ref_model, tokenizer=tokenizer, config=grpo_cfg)
    grpo_rewards: List[float] = []
    for i in range(grpo_steps):
        prompt = all_samples[i % len(all_samples)]["prompt"]
        step_metrics = grpo.train_step(prompt)
        grpo_rewards.append(step_metrics["mean_reward"])

    # --- Stage 3: Live Verification ---
    prefix_cache = LatentPrefixCache(max_entries=8)
    test_prompt = "<forensic> M-Pesa inflow TZS 5,000,000, five debits of TZS 980,000 in 12 min. </forensic>"
    prompt_ids = tokenizer.encode(test_prompt, add_special_tokens=True)
    # Remove trailing EOS from prompt encoding before generation
    if prompt_ids and prompt_ids[-1] == EOS_TOKEN_ID:
        prompt_ids = prompt_ids[:-1]

    generated_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=64,
        temperature=0.0,
        stop_tokens=[EOS_TOKEN_ID],
        prefix_cache=prefix_cache,
    )
    sample_output = tokenizer.decode(generated_ids[len(prompt_ids) :], skip_special_tokens=False)

    elapsed_sec = time.perf_counter() - t0
    results: Dict[str, Any] = {
        "vocab_size": tokenizer.vocab_size,
        "total_params": total_params,
        "initial_sft_loss": sft_losses[0] if sft_losses else 0.0,
        "final_sft_loss": sft_losses[-1] if sft_losses else 0.0,
        "mean_grpo_reward": sum(grpo_rewards) / max(len(grpo_rewards), 1),
        "elapsed_sec": elapsed_sec,
        "sample_prompt": test_prompt,
        "sample_output": sample_output,
    }

    if export_dir:
        export_hf_bundle(
            model=model,
            config=config,
            output_dir=export_dir,
            tokenizer=tokenizer,
            model_name=hf_repo_id,
        )
        results["export_dir"] = export_dir
        if push_to_hf:
            up = upload_hf_bundle(
                bundle_dir=export_dir,
                repo_id=hf_repo_id,
                env_path=env_path,
            )
            results["hf_url"] = up["url"]

    return results
