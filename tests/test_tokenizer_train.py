"""Tests for BPE tokenizer training, merge compression, and save/load persistence."""

import tempfile

from goshawk.tokenizer.bpe import GoshawkTokenizer


def test_bpe_training_compression_and_roundtrip():
    corpus = [
        "TZS 500,000 M-Pesa confirmed from JOHN KWEKA. Salio jipya ni TZS 1,500,000.",
        "TZS 500,000 M-Pesa confirmed from JOHN KWEKA. Salio jipya ni TZS 2,000,000.",
        "<think>Section 40A Evidence Act verification.</think>",
        "<think>Section 40A Evidence Act verification.</think>",
    ]
    raw_tok = GoshawkTokenizer.default()
    trained_tok = GoshawkTokenizer.train(corpus, target_vocab_size=320)

    assert trained_tok.vocab_size > raw_tok.vocab_size
    assert len(trained_tok.merges) > 0

    sample = "<think>TZS 500,000 M-Pesa confirmed from JOHN KWEKA.</think>"
    raw_ids = raw_tok.encode(sample)
    trained_ids = trained_tok.encode(sample)

    # Trained BPE should compress repeated phrases into fewer tokens
    assert len(trained_ids) < len(raw_ids)
    assert trained_tok.decode(trained_ids) == sample


def test_bpe_save_and_load():
    corpus = ["Airtel Money TZS 100,000 confirmed."] * 5
    tok = GoshawkTokenizer.train(corpus, target_vocab_size=300)

    with tempfile.TemporaryDirectory() as tmpdir:
        tok.save(tmpdir)
        loaded = GoshawkTokenizer.load(tmpdir)

        assert loaded.vocab == tok.vocab
        assert loaded.merges == tok.merges
        text = "Airtel Money TZS 100,000 confirmed."
        assert loaded.encode(text) == tok.encode(text)
        assert loaded.decode(loaded.encode(text)) == text
