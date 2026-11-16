"""Tests for MLA LatentPrefixCache and prefix-cached generation."""

import torch

from goshawk.config import GoshawkConfig
from goshawk.model.generation import LatentPrefixCache, generate
from goshawk.model.transformer import GoshawkForCausalLM


def test_prefix_cache_exact_match_and_lru():
    torch.manual_seed(11)
    cfg = GoshawkConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        v_head_dim=16,
        max_seq_len=64,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_dim=32,
        dense_intermediate_dim=32,
        mtp_depth=0,
    )
    model = GoshawkForCausalLM(cfg).eval()
    cache = LatentPrefixCache(max_entries=2)

    system_prefix = [4, 11, 22, 33, 44]
    query1 = system_prefix + [5, 6, 7]
    query2 = system_prefix + [9, 10]

    # Baseline without cache
    baseline1 = generate(model, query1, max_new_tokens=5, temperature=0.0)
    baseline2 = generate(model, query2, max_new_tokens=5, temperature=0.0)

    # First run populates cache for system_prefix
    out1 = generate(
        model,
        query1,
        max_new_tokens=5,
        temperature=0.0,
        prefix_cache=cache,
        cache_prefix_len=len(system_prefix),
    )
    assert out1 == baseline1
    assert cache.stats["misses"] == 1
    assert cache.stats["entries"] == 1

    # Second run reuses system_prefix latent KV cache
    out2 = generate(
        model,
        query2,
        max_new_tokens=5,
        temperature=0.0,
        prefix_cache=cache,
    )
    assert out2 == baseline2
    assert cache.stats["hits"] == 1
