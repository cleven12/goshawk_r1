"""Tests for Goshawk R1 byte-level tokenizer and special tokens."""


from goshawk.tokenizer.bpe import GoshawkTokenizer
from goshawk.tokenizer.vocab import (
    ALERT_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    SPECIAL_TOKENS,
    THINK_END_ID,
    THINK_START_ID,
)


def test_special_tokens_definitions():
    assert "<think>" in SPECIAL_TOKENS
    assert "</think>" in SPECIAL_TOKENS
    assert "<forensic>" in SPECIAL_TOKENS
    assert "<alert>" in SPECIAL_TOKENS
    assert "<tanzania_law>" in SPECIAL_TOKENS


def test_tokenizer_encoding_roundtrip():
    tokenizer = GoshawkTokenizer.default()
    sample_text = "M-Pesa transaction confirmed. Salio jipya ni TZS 250,000."

    tokens = tokenizer.encode(sample_text, add_special_tokens=True)
    assert tokens[0] == BOS_TOKEN_ID
    assert tokens[-1] == EOS_TOKEN_ID

    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    assert decoded == sample_text


def test_special_tokens_preservation():
    tokenizer = GoshawkTokenizer.default()
    text = "<think>Step 1: Check Section 40A.</think><alert>CRITICAL</alert>"

    tokens = tokenizer.encode(text, add_special_tokens=False)
    assert THINK_START_ID in tokens
    assert THINK_END_ID in tokens
    assert ALERT_ID in tokens

    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    assert decoded == text


def test_swahili_text_support():
    tokenizer = GoshawkTokenizer.default()
    swahili_sentence = "Kikosi cha doria kimebaini uvunjaji wa sheria na kuweka vizuizi mara moja."

    tokens = tokenizer.encode(swahili_sentence)
    decoded = tokenizer.decode(tokens)
    assert decoded == swahili_sentence
