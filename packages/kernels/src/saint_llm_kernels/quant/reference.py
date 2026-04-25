"""Torch-eager reference implementation of FP8 quantization.

The reference is intentionally simple and used as the ground truth for any future
Triton/TileLang kernel: every backend must produce bit-identical fp8 bytes and scales
on identical inputs (modulo the documented saturating-clamp behavior below).

Quantization model (per-tensor or per-axis):
    s = amax(|x|) / fmt.fmax              # float32 scale, computed in fp32
    q = clamp(x / s, -fmt.fmax, fmt.fmax) # avoid NaN-on-overflow before the fp8 cast
    fp8 = q.to(fmt.dtype)                 # nearest-even rounding (torch native)

Reconstruction:
    x_hat = fp8.to(float32) * s

Notes:
* Native torch fp8 cast saturates above |x| > fmax to NaN. We pre-clamp so the cast
  is always finite; this matches the standard QAT recipe.
* Scale is always fp32 to keep the reconstruction stable on small magnitudes.
* For per-axis mode, `axis` is the dimension preserved by the scale (e.g. axis=-1
  on a (B, T, D) activation reduces over (B, T) and yields a (D,) scale).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class Fp8Format(Enum):
    """Supported FP8 number formats.

    E4M3 — wider mantissa, smaller range; preferred for activations & weights.
    E5M2 — wider exponent, larger range; preferred for gradients.
    """

    E4M3 = "e4m3"
    E5M2 = "e5m2"

    @property
    def dtype(self) -> torch.dtype:
        return _FORMAT_DTYPE[self]

    @property
    def fmax(self) -> float:
        return torch.finfo(self.dtype).max


_FORMAT_DTYPE: dict[Fp8Format, torch.dtype] = {
    Fp8Format.E4M3: torch.float8_e4m3fn,
    Fp8Format.E5M2: torch.float8_e5m2,
}


@dataclass(frozen=True)
class _ScaleSpec:
    """Resolved scale-reduction plan."""

    reduce_dims: tuple[int, ...]
    keepdim: bool


def _resolve_scale_spec(ndim: int, axis: int | None) -> _ScaleSpec:
    if axis is None:
        return _ScaleSpec(reduce_dims=tuple(range(ndim)), keepdim=False)
    norm_axis = axis % ndim
    return _ScaleSpec(
        reduce_dims=tuple(d for d in range(ndim) if d != norm_axis),
        keepdim=True,
    )


def _amax_scale(x: Tensor, fmax: float, spec: _ScaleSpec) -> Tensor:
    amax = x.detach().abs().to(torch.float32)
    if spec.reduce_dims:
        amax = amax.amax(dim=spec.reduce_dims, keepdim=spec.keepdim)
    return (amax / fmax).clamp(min=torch.finfo(torch.float32).tiny)


def cast_to_fp8(
    x: Tensor,
    fmt: Fp8Format = Fp8Format.E4M3,
    *,
    axis: int | None = None,
    scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Quantize ``x`` to FP8 with absmax scaling.

    Args:
        x: float tensor to quantize. Must be finite.
        fmt: FP8 format (E4M3 by default).
        axis: if None, a single per-tensor scalar scale is used. Otherwise a scale
            is computed per index along ``axis``; the returned scale keeps that axis.
        scale: optional pre-computed scale to bypass amax estimation. Useful for
            paired forward/backward where the forward scale is reused.

    Returns:
        (fp8_tensor, scale) — ``fp8_tensor`` has dtype ``fmt.dtype``, ``scale`` is
        float32. Reconstruction: ``fp8_tensor.to(float32) * scale``.
    """
    spec = _resolve_scale_spec(x.ndim, axis)
    if scale is None:
        scale = _amax_scale(x, fmt.fmax, spec)
    elif axis is not None and scale.ndim != x.ndim:
        scale = scale.reshape([x.shape[axis] if d == axis % x.ndim else 1 for d in range(x.ndim)])

    x_scaled = (x.to(torch.float32) / scale).clamp(-fmt.fmax, fmt.fmax)
    fp8 = x_scaled.to(fmt.dtype)
    return fp8, scale


def dequant_from_fp8(fp8: Tensor, scale: Tensor) -> Tensor:
    """Reconstruct a float32 tensor from ``(fp8, scale)``."""
    return fp8.to(torch.float32) * scale


def fake_quant_fp8(
    x: Tensor,
    fmt: Fp8Format = Fp8Format.E4M3,
    *,
    axis: int | None = None,
) -> Tensor:
    """Quantize-then-dequantize with a straight-through gradient estimator.

    Forward:  ``dequant(quant(x))`` — drops precision to FP8.
    Backward: identity — gradient flows through unchanged.

    Used by FP4/FP8 QAT: the model trains in BF16 but every forward sees FP8 noise.
    """
    fp8, scale = cast_to_fp8(x, fmt=fmt, axis=axis)
    deq = dequant_from_fp8(fp8, scale).to(x.dtype)
    return x + (deq - x).detach()
