"""Tests for Goshawk R1 model architecture, MLA, DeepSeekMoE, and generation."""

import torch

from goshawk.config import GoshawkConfig
from goshawk.model.attention import MultiHeadLatentAttention
from goshawk.model.generation import generate, sample_top_p
from goshawk.model.layers import RMSNorm, RotaryEmbedding, SwiGLU, apply_rotary_pos_emb
from goshawk.model.moe import DeepSeekMoEBlock
from goshawk.model.transformer import GoshawkForCausalLM


def test_config_profiles():
    nano = GoshawkConfig.nano()
    assert nano.d_model == 512
    assert nano.n_layers == 10
    assert nano.n_routed_experts == 6

    micro = GoshawkConfig.micro()
    assert micro.d_model == 1024
    assert micro.n_layers == 16
    assert micro.n_routed_experts == 8

    edge = GoshawkConfig.edge()
    assert edge.d_model == 2048
    assert edge.n_layers == 24


def test_rmsnorm():
    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(2, 8, dim)
    out = norm(x)

    assert out.shape == x.shape
    # Check that variance is approximately 1 along last dimension
    var = torch.mean(out.pow(2), dim=-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2)


def test_swiglu():
    dim = 32
    hidden = 64
    glu = SwiGLU(dim, hidden)
    x = torch.randn(2, 4, dim)
    out = glu(x)
    assert out.shape == (2, 4, dim)


def test_rotary_embedding():
    dim = 16
    max_len = 128
    rope = RotaryEmbedding(dim=dim, max_seq_len=max_len)
    x = torch.randn(2, 4, max_len, dim)

    cos, sin = rope(x, seq_len=16, offset=0)
    assert cos.shape == (16, dim)
    assert sin.shape == (16, dim)

    q = torch.randn(2, 4, 16, dim)
    k = torch.randn(2, 4, 16, dim)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_multi_head_latent_attention():
    # Miniature config for unit testing
    config = GoshawkConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        v_head_dim=16,
        max_seq_len=128,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_dim=64,
        dense_intermediate_dim=64,
    )

    mla = MultiHeadLatentAttention(config)
    x = torch.randn(2, 10, config.d_model)

    out, kv_cache = mla(x)
    assert out.shape == (2, 10, config.d_model)
    assert kv_cache is not None
    assert kv_cache[0].shape == (2, 10, config.kv_lora_rank)

    # Next single token with cache
    x_next = torch.randn(2, 1, config.d_model)
    out_next, new_cache = mla(x_next, kv_cache=kv_cache)
    assert out_next.shape == (2, 1, config.d_model)
    assert new_cache[0].shape == (2, 11, config.kv_lora_rank)


def test_deepseek_moe_block():
    config = GoshawkConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        v_head_dim=16,
        max_seq_len=128,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_dim=64,
        dense_intermediate_dim=64,
    )

    moe = DeepSeekMoEBlock(config)
    x = torch.randn(2, 6, config.d_model)
    out, router_logits = moe(x)

    assert out.shape == (2, 6, config.d_model)
    assert router_logits.shape == (12, config.n_routed_experts)


def test_causal_lm_loss_and_generation():
    config = GoshawkConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        v_head_dim=16,
        max_seq_len=64,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_dim=64,
        dense_intermediate_dim=64,
    )

    model = GoshawkForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = input_ids.clone()

    logits, loss, _ = model(input_ids, labels=labels)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0

    # Test top-p sampling
    next_tok = sample_top_p(logits[:, -1, :], top_p=0.9, temperature=0.7)
    assert next_tok.shape == (2, 1)

    # Test autoregressive generation
    prompt = [10, 15, 20]
    generated = generate(model, prompt, max_new_tokens=4, temperature=0.0)
    assert len(generated) == len(prompt) + 4
    assert generated[: len(prompt)] == prompt
