"""FP4 (E2M1) MXFP4-style emulated quant tests."""

from __future__ import annotations

import pytest
import torch
from saint_llm_kernels.quant import (
    cast_to_fp4_mx,
    dequant_from_fp4_mx,
    fake_quant_fp4_mx,
)
from saint_llm_kernels.quant.fp4 import (
    FP4_E2M1_FMAX,
    FP4_E2M1_LEVELS,
    _quantize_to_fp4_grid,
)


def test_grid_quantize_snaps_to_representable_values() -> None:
    """Every output value of the grid quantizer must be on the FP4 grid."""
    torch.manual_seed(0)
    x = torch.randn(1024) * 3.0
    x = x.clamp(-FP4_E2M1_FMAX, FP4_E2M1_FMAX)
    q = _quantize_to_fp4_grid(x)
    valid = torch.cat([FP4_E2M1_LEVELS, -FP4_E2M1_LEVELS]).unique()
    assert torch.isin(q, valid).all()


def test_grid_quantize_rounds_to_nearest() -> None:
    """Picks the closest grid point — verify on a few hand-checked cases."""
    x = torch.tensor([0.24, 0.26, 1.24, 1.26, 2.49, 2.51, 4.99, 5.01])
    q = _quantize_to_fp4_grid(x)
    expected = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    assert torch.equal(q, expected)


def test_grid_quantize_preserves_sign() -> None:
    x = torch.tensor([-3.7, 3.7, -0.6, 0.6])
    q = _quantize_to_fp4_grid(x)
    assert torch.equal(q.sign(), x.sign())


def test_cast_block_size_must_divide_axis() -> None:
    x = torch.randn(2, 33)  # 33 not divisible by 32
    with pytest.raises(ValueError, match="multiple of block_size"):
        cast_to_fp4_mx(x, block_size=32)


@pytest.mark.parametrize("block_size", [16, 32])
def test_cast_shape_and_scale_shape(block_size: int) -> None:
    """Output shape == input; scale folds the blocked axis."""
    x = torch.randn(4, 64)
    fp4, scale = cast_to_fp4_mx(x, block_size=block_size)
    assert fp4.shape == x.shape
    assert scale.shape == (4, 64 // block_size)


def test_cast_axis_other_than_last() -> None:
    """axis=0 should block-tile the leading dimension."""
    x = torch.randn(64, 8)
    fp4, scale = cast_to_fp4_mx(x, axis=0, block_size=32)
    assert fp4.shape == x.shape
    assert scale.shape == (2, 8)


def test_quantized_values_lie_on_grid() -> None:
    """After dequant, every value must be (FP4 grid level) * (E8M0 power-of-2)."""
    torch.manual_seed(2)
    x = torch.randn(2, 64) * 3.0
    fp4, scale = cast_to_fp4_mx(x, block_size=32)

    n_blocks = scale.shape[-1]
    fp4_blocks = fp4.reshape(2, n_blocks, 32)
    for b in range(2):
        for k in range(n_blocks):
            block_scale = scale[b, k].item()
            block = fp4_blocks[b, k] / block_scale
            on_grid = torch.isin(
                block.abs(),
                FP4_E2M1_LEVELS,
            )
            assert on_grid.all(), f"block ({b}, {k}) has off-grid values: {block.tolist()}"


def test_e8m0_scale_is_power_of_two() -> None:
    """Every per-block scale must be 2^k for integer k (the E8M0 invariant)."""
    torch.manual_seed(3)
    x = torch.randn(8, 128) * 2.5
    _, scale = cast_to_fp4_mx(x, block_size=32)
    log2 = torch.log2(scale)
    assert torch.allclose(log2, log2.round(), atol=1.0e-5)


def test_no_saturation_post_scaling() -> None:
    """Round-up scaling must keep all values on-grid (no clamp loss)."""
    torch.manual_seed(4)
    x = torch.randn(4, 64) * 100.0  # huge magnitudes
    fp4, scale = cast_to_fp4_mx(x, block_size=32)
    # Max representable per-block = scale * FP4_E2M1_FMAX
    max_per_block = scale * FP4_E2M1_FMAX
    fp4_blocked = fp4.reshape(4, scale.shape[-1], 32)
    block_max = fp4_blocked.abs().amax(dim=-1)
    assert (block_max <= max_per_block + 1.0e-5).all()


def test_zero_input_handled() -> None:
    x = torch.zeros(4, 64)
    fp4, scale = cast_to_fp4_mx(x, block_size=32)
    assert torch.isfinite(fp4).all()
    assert torch.isfinite(scale).all()
    assert fp4.abs().max() == 0.0


def test_roundtrip_within_tolerance() -> None:
    """FP4 has 4-bit precision → ~30% relative error is the typical budget."""
    torch.manual_seed(5)
    x = torch.randn(8, 256) * 2.0
    fp4, scale = cast_to_fp4_mx(x, block_size=32)
    deq = dequant_from_fp4_mx(fp4, scale)
    err = (deq - x).abs() / x.abs().clamp(min=1.0e-2)
    # Median (not mean) — outliers around the boundaries can push mean up.
    assert err.median() < 0.30


def test_fake_quant_passes_gradient_through() -> None:
    torch.manual_seed(6)
    x = torch.randn(4, 64, requires_grad=True)
    y = fake_quant_fp4_mx(x, block_size=32)
    y.sum().backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x))


def test_fake_quant_actually_quantizes_forward() -> None:
    torch.manual_seed(7)
    x = torch.randn(4, 64) * 3.0
    y = fake_quant_fp4_mx(x, block_size=32)
    assert not torch.equal(x, y)
    # rounded values are still reasonable
    assert (y - x).abs().max() < x.abs().max() + 1.0


def test_fake_quant_preserves_input_dtype() -> None:
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    y = fake_quant_fp4_mx(x, block_size=32)
    assert y.dtype == torch.bfloat16
