"""mHC carry kernel tests: numerics, shapes, gradient, parity with core MHC."""

from __future__ import annotations

import pytest
import torch
from saint_llm_kernels.mhc import mhc_carry, mhc_carry_reference


def _random_inputs(
    b: int = 2,
    t: int = 16,
    n_hc: int = 4,
    d: int = 64,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    b_l = torch.randn(b, t, n_hc, n_hc, generator=g, dtype=dtype)
    c_l = torch.randn(b, t, n_hc, generator=g, dtype=dtype)
    x = torch.randn(b, t, n_hc, d, generator=g, dtype=dtype)
    inner_out = torch.randn(b, t, d, generator=g, dtype=dtype)
    return b_l, c_l, x, inner_out


def test_reference_shape() -> None:
    """Reference output has the expected (B, T, n_hc, d) shape."""
    b_l, c_l, x, inner_out = _random_inputs()
    out = mhc_carry_reference(b_l, c_l, x, inner_out)
    assert out.shape == x.shape


def test_dispatch_matches_reference_on_cpu() -> None:
    """On CPU, mhc_carry must equal the reference (no fused path active)."""
    b_l, c_l, x, inner_out = _random_inputs()
    expected = mhc_carry_reference(b_l, c_l, x, inner_out)
    actual = mhc_carry(b_l, c_l, x, inner_out)
    assert torch.equal(actual, expected)


def test_carry_term_only_when_c_l_is_zero() -> None:
    """If c_l = 0, the kernel must reduce to einsum(b_l, x)."""
    b_l, c_l, x, inner_out = _random_inputs()
    c_l = torch.zeros_like(c_l)
    out = mhc_carry_reference(b_l, c_l, x, inner_out)
    expected = torch.einsum("bthk,btkd->bthd", b_l, x)
    assert torch.allclose(out, expected, atol=1.0e-6)


def test_write_term_only_when_b_l_is_zero() -> None:
    """If b_l = 0, the kernel must reduce to einsum(c_l, inner_out)."""
    b_l, c_l, x, inner_out = _random_inputs()
    b_l = torch.zeros_like(b_l)
    out = mhc_carry_reference(b_l, c_l, x, inner_out)
    expected = torch.einsum("bth,btd->bthd", c_l, inner_out)
    assert torch.allclose(out, expected, atol=1.0e-6)


def test_gradient_flows_through_all_inputs() -> None:
    """Backward pass must populate grad on every leaf tensor."""
    b_l, c_l, x, inner_out = _random_inputs()
    b_l = b_l.requires_grad_(True)
    c_l = c_l.requires_grad_(True)
    x = x.requires_grad_(True)
    inner_out = inner_out.requires_grad_(True)

    out = mhc_carry(b_l, c_l, x, inner_out)
    out.sum().backward()

    for name, t in [("b_l", b_l), ("c_l", c_l), ("x", x), ("inner_out", inner_out)]:
        assert t.grad is not None, f"{name} has no grad"
        assert torch.isfinite(t.grad).all(), f"{name} has non-finite grad"


@pytest.mark.parametrize(
    "shape",
    [(1, 4, 2, 8), (1, 1, 4, 16), (3, 12, 4, 32), (2, 64, 8, 64)],
)
def test_various_shapes(shape: tuple[int, int, int, int]) -> None:
    b, t, n_hc, d = shape
    b_l, c_l, x, inner_out = _random_inputs(b, t, n_hc, d)
    out = mhc_carry(b_l, c_l, x, inner_out)
    assert out.shape == (b, t, n_hc, d)
    assert torch.isfinite(out).all()


def test_dtype_preservation() -> None:
    """Output dtype matches input dtype (bfloat16 path)."""
    b_l, c_l, x, inner_out = _random_inputs(dtype=torch.bfloat16)
    out = mhc_carry_reference(b_l, c_l, x, inner_out)
    assert out.dtype == torch.bfloat16


@pytest.mark.gpu
def test_dispatch_uses_compiled_path_on_cuda() -> None:
    """On CUDA, the compiled path must produce identical results to reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    b_l, c_l, x, inner_out = _random_inputs()
    cuda = torch.device("cuda")
    b_l, c_l, x, inner_out = (t.to(cuda) for t in (b_l, c_l, x, inner_out))
    expected = mhc_carry_reference(b_l, c_l, x, inner_out)
    actual = mhc_carry(b_l, c_l, x, inner_out)
    assert torch.allclose(actual, expected, atol=1.0e-5)
