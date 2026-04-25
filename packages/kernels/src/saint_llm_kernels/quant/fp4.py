"""FP4 (E2M1) emulated quantization with MXFP4-style block scaling.

Why emulated: torch 2.11 reserves the dtypes ``torch.float4_e2m1fn_x2`` (packed pair)
and ``torch.float8_e8m0fnu`` (power-of-two scale) but does not implement a copy
kernel into FP4 — and Blackwell native FP4 is unavailable on Ada (hpomen) and Apple
silicon. v0.1 stays in float32 with values discretized to the FP4 E2M1 grid; this
is enough for QAT noise injection and for numerics-validating future native kernels.

MXFP4 layout per OCP Microscaling spec:
* Block size 32 (default; NVFP4 = 16).
* Per-block scale stored in E8M0 (power-of-two only, no sign, no mantissa).
* Each element is one of 8 magnitudes by 2 signs.

The 8 representable E2M1 magnitudes are::

    {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

Quantization rounds magnitudes to the nearest grid point (ties→even via torch's
default nearest-even on the comparable arithmetic). Scale is the smallest E8M0 value
such that ``amax(|x|) / scale ≤ 6.0``.
"""

from __future__ import annotations

import torch
from torch import Tensor

FP4_E2M1_FMAX = 6.0
FP4_E2M1_LEVELS: Tensor = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def _quantize_to_fp4_grid(v: Tensor) -> Tensor:
    """Round each element of ``v`` to the nearest FP4 E2M1 representable value.

    Operates on already-scaled values (i.e. caller has divided by per-block scale).
    Preserves sign; tie-breaks via argmin over absolute distance to the grid.
    """
    levels = FP4_E2M1_LEVELS.to(device=v.device, dtype=v.dtype)
    sign = torch.sign(v)
    mag = v.abs().clamp(max=FP4_E2M1_FMAX)
    # (..., 1) vs (8,) → (..., 8) distance to each level.
    distances = (mag.unsqueeze(-1) - levels).abs()
    idx = distances.argmin(dim=-1)
    return sign * levels[idx]


def _compute_e8m0_block_scale(amax: Tensor) -> Tensor:
    """Round-up power-of-two scale such that amax / scale ≤ FP4_E2M1_FMAX.

    Goes through ``torch.float8_e8m0fnu`` so the saved scale is bit-exact with what
    a future native MXFP4 kernel would store.
    """
    raw = (amax / FP4_E2M1_FMAX).clamp(min=torch.finfo(torch.float32).tiny)
    # E8M0 cast already enforces power-of-two with round-to-nearest-even on the
    # exponent. Round up first so we never under-scale and saturate.
    log2_ceil = torch.ceil(torch.log2(raw))
    pow2 = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=amax.device), log2_ceil)
    return pow2.to(torch.float8_e8m0fnu).to(torch.float32)


def cast_to_fp4_mx(
    x: Tensor,
    *,
    block_size: int = 32,
    axis: int = -1,
) -> tuple[Tensor, Tensor]:
    """Block-quantize ``x`` to FP4 E2M1 with MXFP4-style E8M0 block scales.

    Args:
        x: float tensor.
        block_size: number of elements per shared scale (default 32 = MXFP4).
        axis: dimension to block-tile (default -1). The size along ``axis`` must
            be a multiple of ``block_size``.

    Returns:
        (fp4_emulated, scale) — both float32. ``fp4_emulated`` has the same shape
        as ``x`` with values discretized to the FP4 grid. ``scale`` keeps every
        dim of ``x`` except ``axis`` is replaced by ``size_along_axis // block_size``.
    """
    norm_axis = axis % x.ndim
    if x.shape[norm_axis] % block_size != 0:
        raise ValueError(
            f"axis size {x.shape[norm_axis]} must be a multiple of block_size {block_size}",
        )

    # Move target axis to the end for the block reshape; restore after.
    x = x.movedim(norm_axis, -1)
    leading_shape = x.shape[:-1]
    n_blocks = x.shape[-1] // block_size
    x_blocked = x.reshape(*leading_shape, n_blocks, block_size).to(torch.float32)

    amax = x_blocked.abs().amax(dim=-1, keepdim=True)
    scale = _compute_e8m0_block_scale(amax)  # (..., n_blocks, 1)

    x_scaled = x_blocked / scale
    x_q = _quantize_to_fp4_grid(x_scaled)
    fp4_emulated = (x_q * scale).reshape(*leading_shape, n_blocks * block_size)
    fp4_emulated = fp4_emulated.movedim(-1, norm_axis).contiguous()

    scale_out = scale.squeeze(-1).movedim(-1, norm_axis).contiguous()
    return fp4_emulated, scale_out


def dequant_from_fp4_mx(fp4: Tensor, scale: Tensor) -> Tensor:
    """Identity for emulated FP4 — values are already stored at full precision.

    Kept for API symmetry with FP8: downstream code can call ``dequant`` without
    branching on whether the storage is real-packed or emulated.
    """
    del scale
    return fp4.to(torch.float32)


def fake_quant_fp4_mx(
    x: Tensor,
    *,
    block_size: int = 32,
    axis: int = -1,
) -> Tensor:
    """Quantize-then-dequantize FP4 with a straight-through gradient estimator.

    Forward injects FP4 quantization noise; backward passes the gradient through
    unchanged. Used by FP4 QAT — model trains in BF16 but every forward sees FP4
    rounding error.
    """
    fp4, scale = cast_to_fp4_mx(x, block_size=block_size, axis=axis)
    deq = dequant_from_fp4_mx(fp4, scale).to(x.dtype)
    return x + (deq - x).detach()
