"""Byte-level BPE tokenizer with training, persistence, and domain-specific token support."""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from .vocab import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, SPECIAL_TOKENS, UNK_TOKEN_ID


def _bytes_to_token_name(raw: bytes) -> str:
    """Represent a merged byte sequence canonically in the vocabulary."""
    if len(raw) == 1:
        return f"<byte_{raw[0]}>"
    return f"<bpe_{raw.hex()}>"


def _token_name_to_bytes(name: str) -> Optional[bytes]:
    if name.startswith("<byte_") and name.endswith(">"):
        return bytes([int(name[6:-1])])
    if name.startswith("<bpe_") and name.endswith(">"):
        return bytes.fromhex(name[5:-1])
    return None


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
            base_offset = len(self.vocab)
            for b in range(256):
                self.vocab[f"<byte_{b}>"] = base_offset + b

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges: List[Tuple[str, str]] = list(merges or [])
        self.merge_ranks: Dict[Tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self._special_regex = re.compile(
            "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        )

    @classmethod
    def default(cls) -> "GoshawkTokenizer":
        """Instantiate default tokenizer with full special tokens and byte coverage."""
        return cls()

    @classmethod
    def train(cls, texts: List[str], target_vocab_size: int = 512) -> "GoshawkTokenizer":
        """Train byte-level BPE merges on a Swahili/English/forensic text corpus."""
        tok = cls.default()
        if target_vocab_size <= tok.vocab_size:
            return tok

        # Split corpus around special tokens and tokenize into base byte symbols
        corpus_seqs: List[List[str]] = []
        for text in texts:
            parts = tok._special_regex.split(text)
            for part in parts:
                if not part or part in tok.special_tokens_set:
                    continue
                corpus_seqs.append([f"<byte_{b}>" for b in part.encode("utf-8")])

        merges: List[Tuple[str, str]] = []
        while len(tok.vocab) < target_vocab_size:
            pair_counts: Counter = Counter()
            for seq in corpus_seqs:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1

            if not pair_counts:
                break

            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < 2:
                break

            b0 = _token_name_to_bytes(best_pair[0]) or b""
            b1 = _token_name_to_bytes(best_pair[1]) or b""
            merged_name = _bytes_to_token_name(b0 + b1)

            if merged_name not in tok.vocab:
                tok.vocab[merged_name] = len(tok.vocab)
            merges.append(best_pair)

            # Apply merge across corpus
            new_corpus: List[List[str]] = []
            for seq in corpus_seqs:
                new_seq: List[str] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                        new_seq.append(merged_name)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_corpus.append(new_seq)
            corpus_seqs = new_corpus

        return cls(vocab=tok.vocab, merges=merges)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _apply_bpe(self, symbols: List[str]) -> List[str]:
        """Apply learned BPE merges greedily by rank."""
        if not self.merge_ranks or len(symbols) < 2:
            return symbols

        seq = list(symbols)
        while len(seq) >= 2:
            best_rank = None
            best_pair = None
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            b0 = _token_name_to_bytes(best_pair[0]) or b""
            b1 = _token_name_to_bytes(best_pair[1]) or b""
            merged_name = _bytes_to_token_name(b0 + b1)

            new_seq: List[str] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                    new_seq.append(merged_name)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
        return seq

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode string into integer token IDs."""
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(BOS_TOKEN_ID)

        parts = self._special_regex.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens_set:
                tokens.append(self.vocab[part])
            else:
                symbols = [f"<byte_{b}>" for b in part.encode("utf-8")]
                merged = self._apply_bpe(symbols)
                for sym in merged:
                    tokens.append(self.vocab.get(sym, UNK_TOKEN_ID))

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
                continue

            raw_bytes = _token_name_to_bytes(tok_str)
            if raw_bytes is not None:
                byte_buffer.extend(raw_bytes)
            else:
                if byte_buffer:
                    string_parts.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                string_parts.append(tok_str)

        if byte_buffer:
            string_parts.append(byte_buffer.decode("utf-8", errors="replace"))

        return "".join(string_parts)

    def save(self, directory: str) -> None:
        """Persist tokenizer vocabulary and BPE merges to JSON files."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)
        with open(os.path.join(directory, "merges.json"), "w", encoding="utf-8") as f:
            json.dump([list(pair) for pair in self.merges], f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "GoshawkTokenizer":
        """Load a previously saved GoshawkTokenizer from directory."""
        with open(os.path.join(directory, "vocab.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(os.path.join(directory, "merges.json"), "r", encoding="utf-8") as f:
            merges = [tuple(pair) for pair in json.load(f)]
        return cls(vocab=vocab, merges=merges)
