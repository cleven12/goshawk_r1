"""Dataset and tokenization utilities for Goshawk R1 training and forensic fine-tuning."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ..tokenizer.bpe import GoshawkTokenizer
from ..tokenizer.vocab import (
    PAD_TOKEN_ID,
)


@dataclass
class ReasoningSample:
    """A single prompt-reasoning-response sample for R1 training."""

    prompt: str
    thinking: str
    response: str
    task_type: str  # "forensics", "security_event", "statute_verification"
    metadata: Dict[str, Any]


# Synthetic high-quality training examples for edge forensics and security
SYNTHETIC_REASONING_SAMPLES: List[Dict[str, str]] = [
    {
        "prompt": (
            "<forensic> Audit statement: Customer received TZS 5,000,000 via M-Pesa ref QM81299AZ. "
            "Within 12 minutes, five debits of TZS 980,000 each were sent to five distinct phone numbers. "
            "Ending balance is TZS 100,000. Is this rapid dispersal or smurfing? </forensic>"
        ),
        "thinking": (
            "<think>\n"
            "Step 1: Inflow = 5,000,000 TZS.\n"
            "Step 2: Outflow calculation = 5 * 980,000 = 4,900,000 TZS.\n"
            "Step 3: Dispersal ratio = 4,900,000 / 5,000,000 = 0.98 (98%).\n"
            "Step 4: Window is 12 minutes, which is well below the rapid dispersal threshold of 60 minutes.\n"
            "Step 5: Multiple counterparties (5 unique numbers) receiving amounts just under 1,000,000 indicates structuring.\n"
            "Step 6: Under Cybercrimes Act No. 14 of 2015 s.14 and Penal Code Cap 16 s.302, this exhibits classic smurfing.\n"
            "</think>"
        ),
        "response": (
            "CRITICAL ALERT: Structuring and rapid dispersal detected.\n"
            "Total Inflow: TZS 5,000,000\n"
            "Total Outflow: TZS 4,900,000 (98.0% disbursed in 12 minutes)\n"
            "Modus Operandi: Smurfing across 5 distinct recipients to evade mobile money transaction monitoring thresholds.\n"
            "Applicable Law: Penal Code Cap 16 s.302 (Obtaining by False Pretences), Cybercrimes Act 2015 s.14.\n"
            "Admissibility: Electronic records must be certified under Section 40A of Tanzania Evidence Act Cap 6."
        ),
        "task_type": "forensics",
    },
    {
        "prompt": (
            "<security_event> Camera CAM-04 (Rear Gate): At 02:41 AM, two unidentified subjects observed. "
            "Subject A is carrying a pry bar and approaching the warehouse security lock. Subject B is concealing face. </security_event>"
        ),
        "thinking": (
            "<think>\n"
            "Step 1: Timestamp 02:41 AM is after-hours / nocturnal zone.\n"
            "Step 2: Key objects observed: 'pry bar' matches forced entry burglary tool pattern.\n"
            "Step 3: Behavior 'concealing face' at night indicates deliberate evasion of identification.\n"
            "Step 4: Perimeter location 'Rear Gate' is an access control boundary.\n"
            "Step 5: Threat classification must be elevated to CRITICAL due to active burglary tool present.\n"
            "</think>"
        ),
        "response": (
            "<alert> [CRITICAL] Immediate perimeter breach and burglary in progress.\n"
            "Camera: CAM-04 (Rear Gate)\n"
            "Event Type: forced_entry\n"
            "Evidence: pry bar, concealed face\n"
            "Recommended Action: Sound perimeter alarm immediately and dispatch emergency patrol.</alert>"
        ),
        "task_type": "security_event",
    },
    {
        "prompt": (
            "<tanzania_law> Does an M-Pesa SMS confirmation without a Section 40A electronic certificate "
            "suffice as primary evidence in a criminal prosecution in Tanzania? </tanzania_law>"
        ),
        "thinking": (
            "<think>\n"
            "Step 1: Electronic evidence in Tanzania is governed by the Tanzania Evidence Act Cap 6 R.E. 2019.\n"
            "Step 2: Section 40A expressly provides for the admissibility of electronic and digital records.\n"
            "Step 3: Section 40B mandates that electronic records must be accompanied by a certificate of electronic evidence executed by a competent officer.\n"
            "Step 4: Without the statutory certificate under Section 40B, an uncertified SMS screenshot or log is liable to be rejected as inadmissible hearsay or unauthenticated digital output.\n"
            "</think>"
        ),
        "response": (
            "No. Under Section 40A and Section 40B of the Tanzania Evidence Act Cap 6 R.E. 2019, "
            "electronic records such as M-Pesa SMS notifications and server logs require a formal "
            "Certificate of Electronic Evidence certifying the integrity and proper operation of the electronic device. "
            "Without this certificate, the evidence is subject to objection for lack of statutory authentication."
        ),
        "task_type": "statute_verification",
    },
]


class GoshawkReasoningDataset(Dataset):
    """PyTorch Dataset for Goshawk R1 reasoning trajectories."""

    def __init__(
        self,
        samples: Optional[List[Dict[str, str]]] = None,
        tokenizer: Optional[GoshawkTokenizer] = None,
        max_seq_len: int = 1024,
    ):
        self.samples = samples if samples is not None else SYNTHETIC_REASONING_SAMPLES
        self.tokenizer = tokenizer or GoshawkTokenizer.default()
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        full_text = f"{sample['prompt']}\n{sample['thinking']}\n{sample['response']}"

        encoded = self.tokenizer.encode(full_text, add_special_tokens=True)
        if len(encoded) > self.max_seq_len:
            encoded = encoded[: self.max_seq_len]

        input_ids = torch.tensor(encoded, dtype=torch.long)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt": sample["prompt"],
            "response": sample["response"],
            "task_type": sample["task_type"],
        }


def collate_reasoning_batch(
    batch: List[Dict[str, Any]], pad_token_id: int = PAD_TOKEN_ID
) -> Dict[str, Any]:
    """Pad batch of variable length sequences to the maximum length in the batch."""
    max_len = max(len(item["input_ids"]) for item in batch)
    batch_size = len(batch)

    padded_inputs = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_masks = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        padded_inputs[i, :seq_len] = item["input_ids"]
        padded_labels[i, :seq_len] = item["labels"]
        attention_masks[i, :seq_len] = True

    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": attention_masks,
        "prompts": [item["prompt"] for item in batch],
        "responses": [item["response"] for item in batch],
        "task_types": [item["task_type"] for item in batch],
    }
