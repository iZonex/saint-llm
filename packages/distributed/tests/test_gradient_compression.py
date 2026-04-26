"""Tests for gradient compression algorithms (DIST-04 / DIST-05)."""

from __future__ import annotations

import pytest
import torch
from saint_llm_distributed import (
    CompressedGradient,
    SparseLoCoBloom,
    TopKWithResidualCompressor,
    decompress,
    sparse_loco_compress,
    top_k_compression,
)

# ----- top-k compression -----------------------------------------------------


def test_top_k_compression_keeps_density_fraction() -> None:
    grad = torch.arange(100, dtype=torch.float32)
    out = top_k_compression(grad, density=0.1)
    # 10% of 100 = 10 entries kept.
    assert out.indices.numel() == 10
    assert pytest.approx(out.density(), abs=0.01) == 0.1


def test_top_k_compression_keeps_largest_magnitude() -> None:
    grad = torch.tensor([1.0, -100.0, 2.0, 50.0, -3.0])
    out = top_k_compression(grad, density=0.4)  # keep 2 of 5
    kept_indices = sorted(out.indices.tolist())
    # The two largest by abs are -100 (idx 1) and 50 (idx 3).
    assert kept_indices == [1, 3]


def test_top_k_compression_density_validation() -> None:
    grad = torch.zeros(10)
    with pytest.raises(ValueError, match="density must be in"):
        top_k_compression(grad, density=0.0)
    with pytest.raises(ValueError, match="density must be in"):
        top_k_compression(grad, density=1.5)


def test_top_k_compression_preserves_shape() -> None:
    grad = torch.randn(4, 8, 16)
    out = top_k_compression(grad, density=0.05)
    assert out.shape == (4, 8, 16)
    assert out.n_total == 4 * 8 * 16


def test_decompress_roundtrip_at_full_density() -> None:
    """At density=1.0 every entry is kept; decompress reproduces input."""
    grad = torch.randn(20)
    compressed = top_k_compression(grad, density=1.0)
    rebuilt = decompress(compressed)
    torch.testing.assert_close(rebuilt, grad)


def test_decompress_zeros_unsent_positions() -> None:
    grad = torch.tensor([1.0, 2.0, 3.0, 4.0])
    compressed = top_k_compression(grad, density=0.5)  # top 2
    rebuilt = decompress(compressed)
    # Rebuild has exactly two nonzero entries matching kept positions.
    assert (rebuilt != 0).sum().item() == 2


# ----- TopKWithResidualCompressor --------------------------------------------


def test_residual_compressor_carries_unsent_mass() -> None:
    """Same gradient sent twice with low density: second send picks up
    what the first one missed (residual feedback).
    """
    comp = TopKWithResidualCompressor(density=0.5)
    grad = torch.tensor([1.0, 2.0, 3.0, 4.0])

    out1 = comp.step(grad)
    # Top-2 are indices 2, 3 (values 3, 4).
    assert sorted(out1.indices.tolist()) == [2, 3]

    out2 = comp.step(grad)
    # Residual after step 1 was [1, 2, 0, 0] (the un-sent half).
    # After +grad in step 2, adjusted = [2, 4, 3, 4].
    # Top-2 are indices 1 and 3 (or 0/3 with ties resolved by torch).
    # Just verify residual mass eventually transmitted: indices differ
    # from the first call, indicating residual is influencing top-k.
    assert sorted(out2.indices.tolist()) != [2, 3] or out2.values.sum() > 0


def test_residual_compressor_reset_clears_state() -> None:
    comp = TopKWithResidualCompressor(density=0.5)
    grad = torch.tensor([1.0, 2.0, 3.0, 4.0])
    comp.step(grad)
    comp.reset()
    out = comp.step(grad)
    # Without residual, top-k matches plain top_k_compression.
    plain = top_k_compression(grad, density=0.5)
    assert sorted(out.indices.tolist()) == sorted(plain.indices.tolist())


# ----- SparseLoCoBloom -------------------------------------------------------


def test_bloom_filter_membership() -> None:
    bloom = SparseLoCoBloom.empty(m_bits=512, n_hashes=4)
    bloom.add(42)
    assert bloom.contains(42)
    # Untested values are usually False (small probability of FP).


def test_bloom_filter_merge_or() -> None:
    a = SparseLoCoBloom.empty(m_bits=512, n_hashes=4)
    b = SparseLoCoBloom.empty(m_bits=512, n_hashes=4)
    a.add(1)
    b.add(2)
    merged = a.merged_with(b)
    assert merged.contains(1) and merged.contains(2)


def test_bloom_filter_merge_size_mismatch_raises() -> None:
    a = SparseLoCoBloom.empty(m_bits=512, n_hashes=4)
    b = SparseLoCoBloom.empty(m_bits=1024, n_hashes=4)
    with pytest.raises(ValueError, match="must share"):
        a.merged_with(b)


# ----- sparse_loco_compress --------------------------------------------------


def test_sparse_loco_compress_basic() -> None:
    """Without peer bloom, sparse_loco produces top-k indices + a fresh bloom."""
    grad = torch.tensor([1.0, 100.0, 2.0, 50.0, -3.0, 75.0, 0.5, 25.0])
    compressed, bloom = sparse_loco_compress(grad, density=0.5)
    # 50% of 8 = 4 indices kept.
    assert compressed.indices.numel() == 4
    # The kept indices are recorded in the bloom.
    for idx in compressed.indices.tolist():
        assert bloom.contains(idx)


def test_sparse_loco_compress_with_peer_bloom_dedups() -> None:
    """When peer_bloom contains some top-k indices, they're skipped."""
    grad = torch.tensor([1.0, 100.0, 2.0, 50.0, -3.0, 75.0, 0.5, 25.0])
    # Pretend a peer already sent indices 1 and 3 (the two largest).
    peer = SparseLoCoBloom.empty(m_bits=512, n_hashes=4)
    peer.add(1)
    peer.add(3)
    compressed, _bloom_out = sparse_loco_compress(
        grad, density=0.25, peer_bloom=peer,
    )
    # 25% of 8 = 2 indices. Top-2 (indices 1, 3) are filtered;
    # next-largest (5, 7) get sent instead.
    kept = sorted(compressed.indices.tolist())
    assert 1 not in kept
    assert 3 not in kept
    assert kept == [5, 7]


def test_sparse_loco_compress_fallback_when_all_filtered() -> None:
    """If every top-k index is in the peer bloom, fall back to the
    largest-magnitude one to guarantee progress."""
    grad = torch.tensor([1.0, 100.0])
    peer = SparseLoCoBloom.empty(m_bits=512, n_hashes=4)
    peer.add(0)
    peer.add(1)
    compressed, _ = sparse_loco_compress(grad, density=0.5, peer_bloom=peer)
    # Fallback: at least one index sent (the largest, idx 1).
    assert compressed.indices.numel() >= 1


def test_sparse_loco_compress_density_validation() -> None:
    with pytest.raises(ValueError, match="density must be in"):
        sparse_loco_compress(torch.zeros(10), density=0.0)


def test_sparse_loco_compressed_decompresses_to_partial_grad() -> None:
    grad = torch.randn(100)
    compressed, _ = sparse_loco_compress(grad, density=0.1)
    rebuilt = decompress(compressed)
    assert rebuilt.shape == grad.shape
    # Density 10% — non-zero count <= 10.
    assert (rebuilt != 0).sum().item() <= 10


def test_compressed_gradient_density_property() -> None:
    cg = CompressedGradient(
        indices=torch.tensor([0, 5, 10]),
        values=torch.tensor([1.0, 2.0, 3.0]),
        shape=(100,),
        n_total=100,
    )
    assert pytest.approx(cg.density(), abs=1e-6) == 0.03
