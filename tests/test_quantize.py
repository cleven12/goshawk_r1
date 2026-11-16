"""Tests for INT8 and INT4 weight quantization of Goshawk R1 models."""

import torch
import torch.nn as nn

from goshawk.config import GoshawkConfig
from goshawk.export.quantize import PackedInt4Linear, QuantizedLinear, quantize_model
from goshawk.model.transformer import GoshawkForCausalLM


def test_quantized_linear_layer():
    in_feat = 64
    out_feat = 128
    float_linear = nn.Linear(in_feat, out_feat, bias=True)

    q_layer = QuantizedLinear.from_float(float_linear, bits=8)
    assert q_layer.weight_quant.dtype == torch.int8
    assert q_layer.weight_quant.shape == (out_feat, in_feat)
    assert q_layer.weight_scale.shape == (out_feat, 1)

    x = torch.randn(2, 4, in_feat)
    out_float = float_linear(x)
    out_quant = q_layer(x)

    # Verify output shapes match
    assert out_quant.shape == out_float.shape

    # Check relative cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        out_float.flatten(), out_quant.flatten(), dim=0
    )
    assert cos_sim.item() > 0.98


def test_full_model_quantization():
    config = GoshawkConfig(
        vocab_size=128,
        d_model=64,
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
    )
    model = GoshawkForCausalLM(config)

    quant_model, stats = quantize_model(model, bits=8)
    assert stats["replaced_layers"] > 0
    assert stats["compression_ratio"] > 1.0

    # Ensure forward pass succeeds on quantized model
    dummy_input = torch.randint(0, config.vocab_size, (1, 4))
    logits, _, _ = quant_model(dummy_input)
    assert logits.shape == (1, 4, config.vocab_size)


def test_packed_int4_linear_layer():
    in_feat = 65  # Odd dimension to exercise nibble padding
    out_feat = 128
    float_linear = nn.Linear(in_feat, out_feat, bias=True)

    q4 = PackedInt4Linear.from_float(float_linear)
    assert q4.weight_packed.dtype == torch.uint8
    assert q4.weight_packed.shape == (out_feat, 33)

    x = torch.randn(2, 4, in_feat)
    out_float = float_linear(x)
    out_q4 = q4(x)

    assert out_q4.shape == out_float.shape
    cos_sim = torch.nn.functional.cosine_similarity(
        out_float.flatten(), out_q4.flatten(), dim=0
    )
    assert cos_sim.item() > 0.96


def test_full_model_int4_quantization():
    config = GoshawkConfig(
        vocab_size=128,
        d_model=64,
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
    )
    model_int8 = GoshawkForCausalLM(config)
    model_int4 = GoshawkForCausalLM(config)
    model_int4.load_state_dict(model_int8.state_dict())

    _, stats8 = quantize_model(model_int8, bits=8)
    quant4, stats4 = quantize_model(model_int4, bits=4)

    assert stats4["compression_ratio"] > stats8["compression_ratio"]

    dummy_input = torch.randint(0, config.vocab_size, (1, 4))
    logits, _, _ = quant4(dummy_input)
    assert logits.shape == (1, 4, config.vocab_size)
