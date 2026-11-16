"""INT8 and true bit-packed INT4 weight quantization routines for edge CPU inference."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.transformer import GoshawkForCausalLM


class QuantizedLinear(nn.Module):
    """
    Per-channel INT8 weight-quantized linear layer.
    Reduces memory footprint by 4x vs FP32 while maintaining high fidelity on CPU edge nodes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.q_min = -(2 ** (bits - 1))
        self.q_max = (2 ** (bits - 1)) - 1

        self.register_buffer(
            "weight_quant",
            torch.zeros((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones((out_features, 1), dtype=torch.float32),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_float(cls, float_layer: nn.Linear, bits: int = 8) -> "QuantizedLinear":
        """Quantize fp32 nn.Linear weights into int8 representation."""
        layer = cls(
            in_features=float_layer.in_features,
            out_features=float_layer.out_features,
            bias=float_layer.bias is not None,
            bits=bits,
        )

        w = float_layer.weight.data
        max_abs = torch.max(torch.abs(w), dim=-1, keepdim=True).values.clamp(min=1e-8)
        scale = max_abs / float(layer.q_max)

        q_weight = torch.clamp(
            torch.round(w / scale), layer.q_min, layer.q_max
        ).to(torch.int8)

        layer.weight_quant.copy_(q_weight)
        layer.weight_scale.copy_(scale)

        if float_layer.bias is not None:
            layer.bias.copy_(float_layer.bias.data)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dequant_w = self.weight_quant.to(x.dtype) * self.weight_scale.to(x.dtype)
        return F.linear(x, dequant_w, self.bias.to(x.dtype) if self.bias is not None else None)


class PackedInt4Linear(nn.Module):
    """
    True 2-weights-per-byte packed INT4 linear layer.
    Packs pairs of signed 4-bit weights [-8, 7] into uint8 nibbles,
    cutting RAM traffic in half vs INT8 (8x smaller than FP32).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.packed_in = (in_features + 1) // 2

        self.register_buffer(
            "weight_packed",
            torch.zeros((out_features, self.packed_in), dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones((out_features, 1), dtype=torch.float32),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_float(cls, float_layer: nn.Linear) -> "PackedInt4Linear":
        """Quantize fp32 nn.Linear weights and pack 2x INT4 values per uint8 byte."""
        layer = cls(
            in_features=float_layer.in_features,
            out_features=float_layer.out_features,
            bias=float_layer.bias is not None,
        )

        w = float_layer.weight.data
        max_abs = torch.max(torch.abs(w), dim=-1, keepdim=True).values.clamp(min=1e-8)
        scale = max_abs / 7.0

        # Quantize to signed [-8, 7], then shift by +8 into unsigned nibble [0, 15]
        q_signed = torch.clamp(torch.round(w / scale), -8, 7).to(torch.int16)
        q_u4 = (q_signed + 8).to(torch.uint8)

        # Pad odd in_features with zero nibble (8 represents 0)
        if layer.in_features % 2 != 0:
            pad_col = torch.full((layer.out_features, 1), 8, dtype=torch.uint8, device=w.device)
            q_u4 = torch.cat([q_u4, pad_col], dim=1)

        low = q_u4[:, 0::2] & 0x0F
        high = (q_u4[:, 1::2] & 0x0F) << 4
        packed = low | high

        layer.weight_packed.copy_(packed)
        layer.weight_scale.copy_(scale)

        if float_layer.bias is not None:
            layer.bias.copy_(float_layer.bias.data)

        return layer

    def unpack_weights(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Unpack uint8 nibbles back into dequantized weight matrix."""
        low = (self.weight_packed & 0x0F).to(torch.int16) - 8
        high = ((self.weight_packed >> 4) & 0x0F).to(torch.int16) - 8

        # Interleave even and odd columns: [out, packed_in, 2] -> [out, packed_in * 2]
        unpacked_int = torch.stack([low, high], dim=-1).view(self.out_features, -1)
        if unpacked_int.shape[1] > self.in_features:
            unpacked_int = unpacked_int[:, : self.in_features]

        return unpacked_int.to(dtype) * self.weight_scale.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dequant_w = self.unpack_weights(dtype=x.dtype)
        return F.linear(x, dequant_w, self.bias.to(x.dtype) if self.bias is not None else None)


def quantize_model(model: GoshawkForCausalLM, bits: int = 8) -> Tuple[GoshawkForCausalLM, Dict[str, float]]:
    """
    Recursively replaces all linear layers with INT8 QuantizedLinear or packed INT4
    PackedInt4Linear layers. Returns the quantized model and compression statistics.
    """
    if bits not in (4, 8):
        raise ValueError(f"Unsupported quantization bit-width: {bits}. Expected 4 or 8.")

    initial_params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    replaced_count = 0

    def _replace_layers(module: nn.Module) -> None:
        nonlocal replaced_count
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name != "lm_head":
                if bits == 4:
                    quantized = PackedInt4Linear.from_float(child)
                else:
                    quantized = QuantizedLinear.from_float(child, bits=8)
                setattr(module, name, quantized)
                replaced_count += 1
            else:
                _replace_layers(child)

    _replace_layers(model)

    final_bytes = 0
    for p in model.parameters():
        final_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        final_bytes += b.numel() * b.element_size()

    compression_ratio = initial_params_bytes / max(final_bytes, 1)

    stats = {
        "bits": float(bits),
        "replaced_layers": float(replaced_count),
        "original_bytes_mb": initial_params_bytes / (1024 * 1024),
        "quantized_bytes_mb": final_bytes / (1024 * 1024),
        "compression_ratio": compression_ratio,
    }
    return model, stats
