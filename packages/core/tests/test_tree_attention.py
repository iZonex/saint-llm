"""Tests for Tree Attention sequence-parallel via online softmax."""

from __future__ import annotations

import math

import pytest
import torch
from saint_llm_core.attention.common import causal_mask
from saint_llm_core.attention.tree_attention import tree_attention


def _dense_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference: standard scaled-dot-product attention."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    # Rows where everything was masked produce NaN; zero them.
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.einsum("bhqk,bhkd->bhqd", probs, v.float()).to(q.dtype)


def test_tree_attention_matches_dense_with_one_chunk() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8) for _ in range(3))
    expected = _dense_attention(q, k, v)
    actual = tree_attention(q, k, v, n_chunks=1)
    assert torch.allclose(actual, expected, atol=1e-4)


def test_tree_attention_matches_dense_with_multiple_chunks() -> None:
    """Splitting K/V into N chunks should produce the same result."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8) for _ in range(3))
    expected = _dense_attention(q, k, v)
    for n_chunks in (2, 4, 8, 16):
        actual = tree_attention(q, k, v, n_chunks=n_chunks)
        assert torch.allclose(actual, expected, atol=1e-4), (
            f"mismatch at n_chunks={n_chunks}"
        )


def test_tree_attention_with_causal_mask() -> None:
    torch.manual_seed(0)
    t = 12
    q, k, v = (torch.randn(1, 2, t, 8) for _ in range(3))
    mask = causal_mask(t, t)
    expected = _dense_attention(q, k, v, mask=mask)
    actual = tree_attention(q, k, v, n_chunks=4, mask=mask)
    assert torch.allclose(actual, expected, atol=1e-4)


def test_tree_attention_handles_uneven_chunk_split() -> None:
    """T_k that doesn't divide evenly into n_chunks still produces correct output."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 13, 8)  # not divisible by n_chunks=4
    v = torch.randn(1, 2, 13, 8)
    expected = _dense_attention(q, k, v)
    actual = tree_attention(q, k, v, n_chunks=4)
    assert torch.allclose(actual, expected, atol=1e-4)


def test_tree_attention_zeroes_fully_masked_rows() -> None:
    """A row with no visible keys gets zero output (no NaN)."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 6, 8)
    v = torch.randn(1, 1, 6, 8)
    # Row 0 has no visible keys; rows 1..3 see everything.
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[0] = False
    out = tree_attention(q, k, v, n_chunks=2, mask=mask)
    assert torch.equal(out[0, 0, 0], torch.zeros(8))
    # Other rows are non-trivial.
    assert not torch.equal(out[0, 0, 1], torch.zeros(8))


def test_tree_attention_rejects_non_4d() -> None:
    with pytest.raises(ValueError, match=r"\(B, H, T, D\)"):
        tree_attention(torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 4, 8))


def test_tree_attention_rejects_kv_shape_mismatch() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 7)  # wrong head_dim
    with pytest.raises(ValueError, match="must share shape"):
        tree_attention(q, k, v)


def test_tree_attention_rejects_qk_batch_mismatch() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(2, 2, 4, 8)
    v = torch.randn(2, 2, 4, 8)
    with pytest.raises(ValueError, match=r"\(B, H\)"):
        tree_attention(q, k, v)


def test_tree_attention_zero_n_chunks_raises() -> None:
    q = torch.randn(1, 1, 2, 4)
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 4)
    with pytest.raises(ValueError, match="must be positive"):
        tree_attention(q, k, v, n_chunks=0)


def test_tree_attention_grad_flows() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 8, requires_grad=True)
    k = torch.randn(1, 2, 4, 8, requires_grad=True)
    v = torch.randn(1, 2, 4, 8, requires_grad=True)
    out = tree_attention(q, k, v, n_chunks=2)
    out.sum().backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()
    assert k.grad is not None
    assert v.grad is not None


def test_tree_attention_dtype_preserved() -> None:
    q = torch.randn(1, 1, 4, 8, dtype=torch.float32)
    k = torch.randn(1, 1, 4, 8, dtype=torch.float32)
    v = torch.randn(1, 1, 4, 8, dtype=torch.float32)
    out = tree_attention(q, k, v)
    assert out.dtype == torch.float32


def test_tree_attention_chunks_larger_than_seq_clamps() -> None:
    """n_chunks > T_k effectively becomes per-token chunks (chunk_size=1)."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    actual = tree_attention(q, k, v, n_chunks=100)
    expected = _dense_attention(q, k, v)
    assert torch.allclose(actual, expected, atol=1e-4)
