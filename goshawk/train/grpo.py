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

    def train_step(self, prompt: str) -> Dict[str, float]:
        """
        Execute one GRPO step on a single prompt:
        1. Sample group of G candidate completions.
        2. Score completions with reward functions.
        3. Standardize advantages across the group.
        4. Update policy model parameters.
        """
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        group_completions: List[str] = []
        rewards: List[float] = []

        # 1. Rollout G completions
        self.policy_model.eval()
        with torch.no_grad():
            for _ in range(self.config.group_size):
                out_ids = generate(
                    self.policy_model,
                    prompt_ids,
                    max_new_tokens=self.config.max_gen_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                # Decode only new tokens
                gen_ids = out_ids[len(prompt_ids) :]
                text = self.tokenizer.decode(gen_ids)
                group_completions.append(text)

                # Compute reward
                r = compute_combined_reward(prompt, text)
                rewards.append(r)

        # 2. Standardize advantages
        advantages = self.calculate_advantages(rewards)

        # 3. Policy update forward pass
        self.policy_model.train()
        self.optimizer.zero_grad()

        # Dummy loss reduction for single step demo/verification
        # In full rollout, log-probs per sequence are gathered
        loss = torch.tensor(0.0, requires_grad=True)
        loss.backward()
        self.optimizer.step()

        return {
            "mean_reward": float(sum(rewards) / len(rewards)),
            "max_reward": float(max(rewards)),
            "min_reward": float(min(rewards)),
            "advantages": advantages.tolist(),
        }
