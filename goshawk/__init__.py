"""Goshawk R1 - Edge Security & Legal Forensics Reasoning Model."""

from .config import GoshawkConfig
from .tokenizer.bpe import GoshawkTokenizer

__version__ = "0.1.0"

__all__ = [
    "GoshawkBlock",
    "GoshawkConfig",
    "GoshawkForCausalLM",
    "GoshawkModel",
    "GoshawkTokenizer",
    "__version__",
    "generate",
    "sample_top_p",
]


def __getattr__(name: str):
    if name in ("GoshawkBlock", "GoshawkForCausalLM", "GoshawkModel"):
        from .model import transformer

        return getattr(transformer, name)
    if name in ("generate", "sample_top_p"):
        from .model import generation

        return getattr(generation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
