"""DeepSeek-R1 style Group Relative Policy Optimization (GRPO) training loop.

GRPO eliminates the need for an external Critic network by sampling a group
of candidate reasoning paths for each query and standardizing rewards within
the group:
    A_i = (r_i - mean(r)) / (std(r) + eps)
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..model.generation import generate
from ..model.transformer import GoshawkForCausalLM
from ..tokenizer.bpe import GoshawkTokenizer


@dataclass
class GRPOTrainingConfig:
    """Hyperparameters for GRPO reinforcement learning."""

    group_size: int = 4
    clip_epsilon: float = 0.2
    kl_weight: float = 0.04
    temperature: float = 0.8
    top_p: float = 0.95
    max_gen_tokens: int = 128
    learning_rate: float = 1e-5
    weight_decay: float = 0.01


# --- Rule-Based Reward Functions for Edge Forensics & Security ---


def reward_format(completion: str) -> float:
    """
    Rewards correct reasoning structure:
    Must contain valid <think>...</think> tags with non-empty content inside and outside.
    """
    pattern = r"<think>(.*?)</think>(.+)"
    match = re.search(pattern, completion, re.DOTALL)
    if not match:
        # Check if at least tags exist
        if "<think>" in completion and "</think>" in completion:
            return 0.5
        return -1.0

    think_content = match.group(1).strip()
    response_content = match.group(2).strip()

    if len(think_content) > 10 and len(response_content) > 10:
        return 1.0
    return 0.2


def reward_forensic_math(prompt: str, completion: str) -> float:
    """
    Checks if numbers cited in reasoning match transaction quantities in the prompt.
    E.g. Inflow vs Outflow sums.
    """
    prompt_numbers = set(re.findall(r"\b\d[\d,]*\b", prompt))
    if not prompt_numbers:
        return 0.0

    # Clean commas
    clean_prompt_nums = {n.replace(",", "") for n in prompt_numbers if len(n) > 2}

    think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
    if not think_match:
        return 0.0

    think_text = think_match.group(1)
    think_numbers = {n.replace(",", "") for n in re.findall(r"\b\d[\d,]*\b", think_text)}

    # Check overlap
    overlap = clean_prompt_nums.intersection(think_numbers)
    if len(overlap) >= 2:
        return 1.0
    elif len(overlap) == 1:
        return 0.5
    return 0.0


def reward_legal_statute(completion: str) -> float:
    """
    Rewards accurate citation of East African criminal and digital evidence statutes.
    """
    statute_patterns = [
        r"section\s*40[ab]?",
        r"penal\s*code",
        r"section\s*(?:265|271|302|311)",
        r"cybercrimes\s*act",
        r"evidence\s*act",
    ]
    score = 0.0
    completion_lower = completion.lower()
    for pat in statute_patterns:
        if re.search(pat, completion_lower):
            score += 0.25

    return min(score, 1.0)


def reward_security_threat(prompt: str, completion: str) -> float:
    """
    Rewards accurate threat classification based on surveillance event cues.
    """
    prompt_lower = prompt.lower()
    completion_lower = completion.lower()

    is_lethal = bool(re.search(r"(gun|pistol|knife|machete|crowbar|pry\s*bar)", prompt_lower))
    is_violent = bool(re.search(r"(altercation|assault|fighting)", prompt_lower))

    if is_lethal:
        if "critical" in completion_lower:
            return 1.0
        elif "high" in completion_lower:
            return 0.4
        return -0.5
    elif is_violent:
        if "high" in completion_lower or "critical" in completion_lower:
            return 1.0
        return -0.2

    return 0.0


def compute_combined_reward(prompt: str, completion: str) -> float:
    """Combined weighted reward for DeepSeek-R1 GRPO step."""
    r_fmt = reward_format(completion)
    r_math = reward_forensic_math(prompt, completion)
    r_law = reward_legal_statute(completion)
    r_sec = reward_security_threat(prompt, completion)

    # Total reward combination
    return (0.35 * r_fmt) + (0.25 * r_math) + (0.20 * r_law) + (0.20 * r_sec)


# --- GRPO Policy Optimizer ---


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer for Goshawk R1.
    """

    def __init__(
        self,
        policy_model: GoshawkForCausalLM,
        ref_model: Optional[GoshawkForCausalLM] = None,
        tokenizer: Optional[GoshawkTokenizer] = None,
        config: Optional[GRPOTrainingConfig] = None,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer or GoshawkTokenizer.default()
        self.config = config or GRPOTrainingConfig()

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def calculate_advantages(self, rewards: List[float], eps: float = 1e-6) -> torch.Tensor:
        """
        Normalize rewards within a sampled group to compute relative advantages:
            A_i = (r_i - mean(r)) / (std(r) + eps)
        """
        r_tensor = torch.tensor(rewards, dtype=torch.float32)
        if len(rewards) <= 1:
            return torch.zeros_like(r_tensor)

        mean = r_tensor.mean()
        std = r_tensor.std(unbiased=False)

        if std < eps:
            return torch.zeros_like(r_tensor)

        return (r_tensor - mean) / (std + eps)

    def compute_grpo_loss(
        self,
        log_probs_policy: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
        log_probs_ref: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute clipped surrogate GRPO policy loss and KL divergence penalty.
        """
        # Probability ratio
        ratio = torch.exp(log_probs_policy - log_probs_old)

        # Clipped surrogate objective
        unclipped_loss = ratio * advantages
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        )
        clipped_loss = clipped_ratio * advantages
        policy_loss = -torch.min(unclipped_loss, clipped_loss).mean()

        # Token-level KL penalty against reference model
        kl_div = torch.tensor(0.0, device=log_probs_policy.device)
        if log_probs_ref is not None:
            # KL(pi || ref) approx (log_probs_policy - log_probs_ref)
            kl_div = (torch.exp(log_probs_ref - log_probs_policy) - (log_probs_ref - log_probs_policy) - 1.0).mean()

        total_loss = policy_loss + (self.config.kl_weight * kl_div)

        metrics = {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "kl_divergence": float(kl_div.item()),
            "mean_ratio": float(ratio.mean().item()),
        }
        return total_loss, metrics

    def _completion_log_prob(
        self,
        model: GoshawkForCausalLM,
        full_ids: List[int],
        prompt_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute mean per-token log-probability of generated completion tokens."""
        seq = torch.tensor([full_ids], dtype=torch.long, device=device)
        logits, _, _ = model(seq)
        shift_logits = logits[:, prompt_len - 1 : -1, :]
        target_ids = seq[:, prompt_len:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return token_lp.mean()

    def train_step(self, prompt: str) -> Dict[str, float]:
        """
        Execute one GRPO step on a single prompt:
        1. Sample group of G candidate completions and record old/ref log-probs.
        2. Score completions with deterministic rule-based verifiers.
        3. Standardize advantages across the group.
        4. Update policy model parameters via clipped GRPO + KL objective.
        """
        dev = next(self.policy_model.parameters()).device
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids)

        rollout_ids: List[List[int]] = []
        old_lps: List[torch.Tensor] = []
        ref_lps: List[torch.Tensor] = []
        rewards: List[float] = []

        # 1. Rollout G completions
        self.policy_model.eval()
        if self.ref_model is not None:
            self.ref_model.eval()

        with torch.no_grad():
            for _ in range(self.config.group_size):
                out_ids = generate(
                    self.policy_model,
                    prompt_ids,
                    max_new_tokens=self.config.max_gen_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    device=dev,
                )
                gen_ids = out_ids[prompt_len:]
                text = self.tokenizer.decode(gen_ids)
                rewards.append(compute_combined_reward(prompt, text))
                rollout_ids.append(out_ids)

                if len(gen_ids) > 0:
                    old_lps.append(self._completion_log_prob(self.policy_model, out_ids, prompt_len, dev))
                    if self.ref_model is not None:
                        ref_lps.append(self._completion_log_prob(self.ref_model, out_ids, prompt_len, dev))

        # 2. Standardize advantages
        advantages = self.calculate_advantages(rewards).to(dev)

        # 3. Policy update forward pass with real token log-prob gradients
        self.policy_model.train()
        self.optimizer.zero_grad()

        policy_lps = [
            self._completion_log_prob(self.policy_model, seq_ids, prompt_len, dev)
            for seq_ids in rollout_ids
            if len(seq_ids) > prompt_len
        ]

        loss_metrics: Dict[str, float] = {"total_loss": 0.0, "policy_loss": 0.0, "kl_divergence": 0.0}
        if policy_lps:
            lp_policy_t = torch.stack(policy_lps)
            lp_old_t = torch.stack(old_lps)
            lp_ref_t = torch.stack(ref_lps) if ref_lps else None

            loss, loss_metrics = self.compute_grpo_loss(
                lp_policy_t, lp_old_t, advantages[: len(policy_lps)], log_probs_ref=lp_ref_t
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()

        return {
            "mean_reward": float(sum(rewards) / len(rewards)),
            "max_reward": float(max(rewards)),
            "min_reward": float(min(rewards)),
            "total_loss": loss_metrics["total_loss"],
            "policy_loss": loss_metrics["policy_loss"],
            "kl_divergence": loss_metrics["kl_divergence"],
        }
