"""Symmetric per-channel int8 quantization (LiteRT-style, zero-point = 0).

Only weights are quantized; activations stay float32 on device (Xtensa has an
FPU; the flash saving is 4x and each output row costs one extra multiply).
The dequantized mirrors below are what the C engine is numerically equivalent
to, so host and firmware agree within a few ULPs of float32 accumulation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

QMAX = 127.0


def quantize_per_channel(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """w: [out, in] (or [rows, dim]) -> (q int8 [out, in], scale float [out])."""
    assert w.dim() == 2
    max_abs = w.detach().abs().amax(dim=1)
    scale = torch.clamp(max_abs / QMAX, min=1e-12)
    q = torch.round(w.detach() / scale.unsqueeze(1)).clamp(-QMAX, QMAX)
    return q.to(torch.int8), scale.float()


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.unsqueeze(1)


def fake_quant(w: torch.Tensor) -> torch.Tensor:
    """Round-trip through int8; used to build the firmware-equivalent mirror."""
    q, s = quantize_per_channel(w)
    return dequantize(q, s)


class QuantTable:
    """Holds every int8 tensor + scale for the exported model."""

    def __init__(self) -> None:
        self.q: Dict[str, torch.Tensor] = {}        # int8 [out, in]
        self.scale: Dict[str, torch.Tensor] = {}    # float [out]
        self.f32: Dict[str, torch.Tensor] = {}      # biases, LayerNorms

    def add_linear(self, name: str, weight: torch.Tensor, bias=None) -> None:
        q, s = quantize_per_channel(weight)
        self.q[name] = q
        self.scale[name] = s
        if bias is not None:
            self.f32[f"{name}.bias"] = bias.detach().float().clone()

    def add_f32(self, name: str, tensor: torch.Tensor) -> None:
        self.f32[name] = tensor.detach().float().clone()

    @property
    def int8_bytes(self) -> int:
        return int(sum(t.numel() for t in self.q.values()))

    @property
    def float_bytes(self) -> int:
        return int(4 * sum(t.numel() for t in self.f32.values()) +
                   4 * sum(t.numel() for t in self.scale.values()))


def resolve(module, dotted: str):
    obj = module
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def apply_dequant_mirror(model, table: QuantTable, mapping: Dict[str, str]) -> None:
    """Replace each registered [out, in] weight in `model` with its int8
    round-trip so the PyTorch mirror evaluates exactly the dequantized math."""
    with torch.no_grad():
        for dotted, qname in mapping.items():
            mod = resolve(model, dotted)
            mod.weight.copy_(dequantize(table.q[qname], table.scale[qname]))
