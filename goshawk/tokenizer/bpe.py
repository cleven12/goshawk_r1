"""Byte-level tokenizer with domain-specific token support for Goshawk R1."""

import re
from typing import Dict, List, Optional, Set, Tuple

from .vocab import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, SPECIAL_TOKENS, UNK_TOKEN_ID


class GoshawkTokenizer:
    """
    Byte-level BPE tokenizer for Goshawk R1.
    Handles Swahili, English, financial numbers, and specialized forensic & security tokens.
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
    ):
        self.special_tokens: List[str] = list(SPECIAL_TOKENS)
        self.special_tokens_set: Set[str] = set(self.special_tokens)

        if vocab is not None:
            self.vocab = dict(vocab)
        else:
            self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
            # Populate basic byte tokens (256 bytes)
            base_offset = len(self.vocab)
            for b in range(256):
                self.vocab[f"<byte_{b}>"] = base_offset + b

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges or []
        self._special_regex = re.compile(
            "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        )

    @classmethod
    def default(cls) -> "GoshawkTokenizer":
        """Instantiate default tokenizer with full special tokens and byte coverage."""
        return cls()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode string into integer token IDs."""
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(BOS_TOKEN_ID)

        # Split text into chunks delimited by special tokens
        parts = self._special_regex.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens_set:
                tokens.append(self.vocab[part])
            else:
                # Encode raw bytes
                b_bytes = part.encode("utf-8")
                for b in b_bytes:
                    tok_name = f"<byte_{b}>"
                    if tok_name in self.vocab:
                        tokens.append(self.vocab[tok_name])
                    else:
                        tokens.append(UNK_TOKEN_ID)

        if add_special_tokens:
            tokens.append(EOS_TOKEN_ID)

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode integer token IDs back into string."""
        byte_buffer: bytearray = bytearray()
        string_parts: List[str] = []

        for tid in token_ids:
            if tid == PAD_TOKEN_ID or tid not in self.inverse_vocab:
                continue
            tok_str = self.inverse_vocab[tid]

            if tok_str in self.special_tokens_set:
                if byte_buffer:
                    string_parts.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                if not skip_special_tokens:
                    string_parts.append(tok_str)
            elif tok_str.startswith("<byte_") and tok_str.endswith(">"):
                b_val = int(tok_str[6:-1])
                byte_buffer.append(b_val)
            else:
                if byte_buffer:
                    string_parts.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                string_parts.append(tok_str)

        if byte_buffer:
            string_parts.append(byte_buffer.decode("utf-8", errors="replace"))

        return "".join(string_parts)
