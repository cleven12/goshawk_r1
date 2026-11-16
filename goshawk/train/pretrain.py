"""Causal language model pretraining and fine-tuning engine for Goshawk R1."""

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..model.transformer import GoshawkForCausalLM
from .dataset import GoshawkReasoningDataset, collate_reasoning_batch


@dataclass
class PretrainConfig:
    """Hyperparameters for standard next-token causal pretraining."""

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.05
    max_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    checkpoint_dir: str = "./checkpoints"


def get_cosine_lr(step: int, total_steps: int, warmup_steps: int, lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return lr * (step + 1) / max(warmup_steps, 1)
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


class GoshawkTrainer:
    """Orchestrates pretraining and supervised fine-tuning loops."""

    def __init__(
        self,
        model: GoshawkForCausalLM,
        train_dataset: GoshawkReasoningDataset,
        eval_dataset: Optional[GoshawkReasoningDataset] = None,
        config: Optional[PretrainConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or PretrainConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_reasoning_batch,
        )
        self.eval_loader = (
            DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_reasoning_batch,
            )
            if eval_dataset is not None
            else None
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one complete training epoch over the dataset."""
        self.model.train()
        total_loss = 0.0
        step_count = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits, loss, _ = self.model(input_ids, labels=labels)
            if loss is None:
                continue

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()

            total_loss += float(loss.item())
            step_count += 1

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        avg_loss = total_loss / max(step_count, 1)
        return {"epoch": epoch, "train_loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20.0))}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Compute validation loss and perplexity on eval dataset."""
        if not self.eval_loader:
            return {"eval_loss": 0.0, "perplexity": 1.0}

        self.model.eval()
        total_loss = 0.0
        step_count = 0

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            _, loss, _ = self.model(input_ids, labels=labels)
            if loss is not None:
                total_loss += float(loss.item())
                step_count += 1

        avg_loss = total_loss / max(step_count, 1)
        return {"eval_loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20.0))}

    def save_checkpoint(self, path: str) -> None:
        """Serialize model weights and hyperparameters."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "config": self.model.config,
                "state_dict": self.model.state_dict(),
            },
            path,
        )
