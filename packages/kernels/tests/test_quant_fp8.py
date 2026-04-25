"""FP8 reference quant tests: roundtrip, scale modes, saturation, QAT gradient."""

from __future__ import annotations

import pytest
import torch
from saint_llm_kernels.quant import (
    Fp8Format,
    cast_to_fp8,
    dequant_from_fp8,
    fake_quant_fp8,
)


@pytest.mark.parametrize("fmt", [Fp8Format.E4M3, Fp8Format.E5M2])
def test_fmt_dtype_and_fmax(fmt: Fp8Format) -> None:
    """Format wrapper exposes the right torch dtype and finfo max."""
    assert fmt.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert fmt.fmax == torch.finfo(fmt.dtype).max
    if fmt is Fp8Format.E4M3:
        assert fmt.fmax == 448.0
    else:
        assert fmt.fmax == 57344.0


@pytest.mark.parametrize("fmt", [Fp8Format.E4M3, Fp8Format.E5M2])
def test_per_tensor_roundtrip_within_tolerance(fmt: Fp8Format) -> None:
    """A representative activation tensor roundtrips within FP8 quantization noise."""
    torch.manual_seed(0)
    x = torch.randn(8, 64, 128) * 3.0  # mostly within E4M3 range
    fp8, scale = cast_to_fp8(x, fmt=fmt)

    assert fp8.dtype == fmt.dtype
    assert scale.shape == ()
    assert scale.dtype == torch.float32

    deq = dequant_from_fp8(fp8, scale)
    err = (deq - x).abs() / x.abs().clamp(min=1.0e-3)
    # E4M3: 3-bit mantissa → ~12% relative; E5M2: 2-bit → ~25%. Use a loose budget.
    budget = 0.20 if fmt is Fp8Format.E4M3 else 0.40
    assert err.median() < budget


def test_per_tensor_scale_is_scalar() -> None:
    x = torch.randn(2, 4, 8)
    _, scale = cast_to_fp8(x)
    assert scale.shape == ()


def test_per_axis_scale_shape_preserves_axis() -> None:
    """axis=-1 on a (B, T, D) tensor yields a (1, 1, D) scale."""
    x = torch.randn(3, 5, 7) * 10.0
    fp8, scale = cast_to_fp8(x, axis=-1)
    assert fp8.shape == x.shape
    assert scale.shape == (1, 1, 7)
    deq = dequant_from_fp8(fp8, scale)
    assert deq.shape == x.shape


def test_per_axis_scales_differ_per_index() -> None:
    """Per-channel scale should track per-channel magnitude."""
    x = torch.zeros(4, 6)
    x[:, 0] = 0.1
    x[:, 1] = 100.0
    _, scale = cast_to_fp8(x, axis=-1)
    assert scale[0, 0].item() < scale[0, 1].item()


def test_saturation_clamp_does_not_produce_nan() -> None:
    """Inputs above fmax must clamp; the cast must never produce NaN."""
    x = torch.tensor([0.0, 100.0, 1000.0, 1e6, -1e6])
    fp8, _ = cast_to_fp8(x, fmt=Fp8Format.E4M3)
    deq = dequant_from_fp8(fp8, torch.tensor(1.0))
    assert torch.isfinite(deq).all()


def test_zero_input_is_handled() -> None:
    """All-zero input must not produce NaN scales or values."""
    x = torch.zeros(4, 8)
    fp8, scale = cast_to_fp8(x)
    deq = dequant_from_fp8(fp8, scale)
    assert torch.isfinite(scale).all()
    assert torch.isfinite(deq).all()
    assert deq.abs().max() == 0.0


def test_supplied_scale_is_used_verbatim() -> None:
    """Passing scale= bypasses amax estimation."""
    x = torch.randn(4, 8) * 2.0
    custom_scale = torch.tensor(1.0e-2)
    _fp8, scale = cast_to_fp8(x, scale=custom_scale)
    assert torch.equal(scale, custom_scale)


def test_fake_quant_passes_gradient_through() -> None:
    """fake_quant must be lossy on values but identity on gradients."""
    torch.manual_seed(0)
    x = torch.randn(16, 32, requires_grad=True)
    y = fake_quant_fp8(x, fmt=Fp8Format.E4M3)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x))


def test_fake_quant_actually_quantizes_forward() -> None:
    """fake_quant output must differ from input (not a no-op)."""
    torch.manual_seed(1)
    x = torch.randn(64, 64) * 5.0
    y = fake_quant_fp8(x, fmt=Fp8Format.E4M3)
    assert not torch.equal(x, y)
    # but should be close
    assert (y - x).abs().max() < x.abs().max()


def test_fake_quant_preserves_input_dtype() -> None:
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    y = fake_quant_fp8(x)
    assert y.dtype == torch.bfloat16


def test_e5m2_handles_larger_range_than_e4m3() -> None:
    """E5M2 has fmax≈57344; E4M3 saturates at 448."""
    x = torch.tensor([10000.0, -10000.0])
    fp8_e4m3, scale_e4m3 = cast_to_fp8(x, fmt=Fp8Format.E4M3)
    fp8_e5m2, scale_e5m2 = cast_to_fp8(x, fmt=Fp8Format.E5M2)
    deq_e4m3 = dequant_from_fp8(fp8_e4m3, scale_e4m3)
    deq_e5m2 = dequant_from_fp8(fp8_e5m2, scale_e5m2)
    err_e4m3 = (deq_e4m3 - x).abs().max()
    err_e5m2 = (deq_e5m2 - x).abs().max()
    # Both work because we scale; but E5M2 has finer steps at this range
    # → smaller absolute error post-roundtrip.
    assert err_e5m2 <= err_e4m3 + 1.0
