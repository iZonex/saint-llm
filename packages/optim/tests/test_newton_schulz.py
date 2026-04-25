"""Tests for hybrid Newton-Schulz orthogonalization."""

from __future__ import annotations

import torch

from saint_llm_optim import hybrid_newton_schulz


def test_singular_values_close_to_one_for_random_tall() -> None:
    torch.manual_seed(42)
    M = torch.randn(16, 8)
    UVT = hybrid_newton_schulz(M)
    s = torch.linalg.svdvals(UVT)
    err = (s - 1.0).abs().max().item()
    assert err < 0.05, f"Singular values not close to 1: {s.tolist()}"


def test_singular_values_close_to_one_for_random_wide() -> None:
    torch.manual_seed(42)
    M = torch.randn(8, 16)
    UVT = hybrid_newton_schulz(M)
    s = torch.linalg.svdvals(UVT)
    err = (s - 1.0).abs().max().item()
    assert err < 0.05, f"Singular values not close to 1: {s.tolist()}"


def test_preserves_shape() -> None:
    for shape in [(8, 16), (16, 8), (32, 32), (4, 64), (64, 4)]:
        M = torch.randn(*shape)
        out = hybrid_newton_schulz(M)
        assert out.shape == M.shape


def test_batched_input() -> None:
    M = torch.randn(4, 8, 16)
    out = hybrid_newton_schulz(M)
    assert out.shape == M.shape
    s = torch.linalg.svdvals(out)
    err = (s - 1.0).abs().max().item()
    assert err < 0.05


def test_invariant_to_global_scale() -> None:
    """NS uses Frobenius normalization first → output is scale-invariant in input."""
    torch.manual_seed(0)
    M = torch.randn(8, 16)
    out_a = hybrid_newton_schulz(M)
    out_b = hybrid_newton_schulz(M * 1000.0)
    assert torch.allclose(out_a, out_b, atol=1.0e-4)


def test_orthogonal_property_M_Mt_close_to_identity_for_tall() -> None:
    """For tall M (n >= k), UV^T should satisfy (UV^T)^T (UV^T) ≈ I_k."""
    torch.manual_seed(0)
    M = torch.randn(16, 8)
    UVT = hybrid_newton_schulz(M)
    gram = UVT.transpose(-2, -1) @ UVT
    eye = torch.eye(gram.shape[-1])
    err = (gram - eye).abs().max().item()
    assert err < 0.05, f"Gram matrix not close to identity, max err {err}"
