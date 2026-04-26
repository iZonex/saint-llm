"""Gradient compression for low-bandwidth distributed training.

The core algorithmic ingredient that makes decentralized training
viable on commodity internet links. Three algorithms ship here:

* :func:`top_k_compression` — keep only the top-k absolute-magnitude
  entries, zero the rest. Trivial baseline.
* :func:`top_k_with_residual` — top-k plus error feedback (residual
  carry-over). The classic Lin et al. (2018) "Deep Gradient
  Compression" technique. Lossy but unbiased over time.
* :func:`sparse_loco_compress` — Templar SparseLoCo (DIST-05): top-k
  selection + Bloom-filter dedup of indices across pseudo-gradients.
  Achieves the published 97% gradient compression at no measured
  accuracy loss. The "BTC-organism" enabler.

All three operate on a single tensor; the orchestration layer (DiLoCo
outer loop) calls them per-parameter or per-bucket.

References:
    Lin et al. 2018 "Deep Gradient Compression"
    Templar SparseLoCo (Bittensor SN3, March 2026)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(frozen=True)
class CompressedGradient:
    """Sparse representation of a compressed gradient.

    Attributes:
        indices:       flat indices into the original tensor's flatten().
        values:        the kept magnitudes at those indices.
        shape:         original tensor shape (for decompression).
        n_total:       total elements in the original tensor.
    """

    indices: Tensor
    values: Tensor
    shape: tuple[int, ...]
    n_total: int

    def density(self) -> float:
        """Fraction of original elements retained (0..1, lower = more compression)."""
        if self.n_total == 0:
            return 0.0
        return float(self.indices.numel()) / self.n_total


def _topk_indices_values(grad: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Return the top-k absolute-magnitude (indices, values) over a flat grad."""
    flat = grad.flatten()
    k = max(1, min(k, flat.numel()))
    abs_vals = flat.abs()
    _, top_idx = abs_vals.topk(k, sorted=False)
    top_vals = flat[top_idx]
    return top_idx, top_vals


def top_k_compression(grad: Tensor, *, density: float) -> CompressedGradient:
    """Plain top-k: keep ``density`` fraction of the largest-magnitude entries.

    Args:
        grad:    gradient tensor (any shape).
        density: target density in (0, 1]. ``density=0.03`` keeps 3% of
            entries (97% compression — matches SparseLoCo headline).

    Returns:
        :class:`CompressedGradient` with the kept indices and values.
    """
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1]; got {density}")
    n_total = grad.numel()
    k = max(1, round(density * n_total))
    indices, values = _topk_indices_values(grad, k)
    return CompressedGradient(
        indices=indices, values=values,
        shape=tuple(grad.shape), n_total=n_total,
    )


@dataclass
class TopKWithResidualCompressor:
    """Top-k compressor with error feedback (residual carry-over).

    Maintains a per-parameter residual buffer so the gradient mass not
    sent in step ``t`` is added to step ``t+1``. Over time the
    compression is **unbiased** — every gradient component eventually
    gets transmitted.

    Per-parameter state lives in this object; instantiate one per
    parameter you compress (or one per group).
    """

    density: float
    _residual: Tensor | None = field(default=None, init=False, repr=False)

    def step(self, grad: Tensor) -> CompressedGradient:
        if self._residual is None or self._residual.shape != grad.shape:
            self._residual = torch.zeros_like(grad)
        adjusted = grad + self._residual
        compressed = top_k_compression(adjusted, density=self.density)

        # Update residual: subtract the part that was sent.
        sent = torch.zeros_like(adjusted)
        sent.flatten().scatter_(0, compressed.indices, compressed.values)
        self._residual = adjusted - sent
        return compressed

    def reset(self) -> None:
        self._residual = None


def decompress(compressed: CompressedGradient, *, dtype: torch.dtype | None = None) -> Tensor:
    """Reconstruct a full-size gradient from its sparse representation."""
    out_dtype = dtype if dtype is not None else compressed.values.dtype
    flat = torch.zeros(
        compressed.n_total, dtype=out_dtype, device=compressed.values.device,
    )
    flat.scatter_(0, compressed.indices, compressed.values.to(out_dtype))
    return flat.reshape(compressed.shape)


# ---- SparseLoCo: top-k + Bloom-filter dedup -------------------------------


def _bloom_size_for(n_items: int, false_pos_rate: float = 0.01) -> tuple[int, int]:
    """Recommended (m, k) for ``n_items`` and target false-positive rate."""
    # Standard Bloom optimal: m = -n * ln(p) / (ln(2)^2); k = (m/n) ln(2).
    import math  # noqa: PLC0415

    m = max(64, int(-n_items * math.log(false_pos_rate) / (math.log(2) ** 2)))
    n_hash = max(1, round((m / max(n_items, 1)) * math.log(2)))
    return m, n_hash


def _bloom_hash(idx: int, salt: int, bits: int) -> int:
    """Deterministic hash of an integer index + salt to a bit position."""
    h = hashlib.sha1(f"{salt}:{idx}".encode()).digest()
    return int.from_bytes(h[:8], "big") % bits


@dataclass(frozen=True)
class SparseLoCoBloom:
    """Bloom filter snapshot of indices already transmitted by peers.

    Carries the bit-array + hashing parameters. When merging across
    peers, simply OR the bit-arrays.
    """

    bits: Tensor  # uint8 dtype, length == m_bits / 8
    m_bits: int
    n_hashes: int

    @classmethod
    def empty(cls, *, m_bits: int, n_hashes: int) -> SparseLoCoBloom:
        return cls(
            bits=torch.zeros((m_bits + 7) // 8, dtype=torch.uint8),
            m_bits=m_bits, n_hashes=n_hashes,
        )

    def _bit_set(self, bit_idx: int) -> bool:
        byte = bit_idx // 8
        offset = bit_idx % 8
        return bool(int(self.bits[byte].item()) & (1 << offset))

    def contains(self, idx: int) -> bool:
        return all(
            self._bit_set(_bloom_hash(idx, h, self.m_bits))
            for h in range(self.n_hashes)
        )

    def add(self, idx: int) -> None:
        for h in range(self.n_hashes):
            bit_idx = _bloom_hash(idx, h, self.m_bits)
            byte = bit_idx // 8
            offset = bit_idx % 8
            self.bits[byte] = (
                self.bits[byte].to(torch.int64) | (1 << offset)
            ).to(torch.uint8)

    def merged_with(self, other: SparseLoCoBloom) -> SparseLoCoBloom:
        """OR-merge two bloom filters of the same shape."""
        if self.m_bits != other.m_bits or self.n_hashes != other.n_hashes:
            raise ValueError("bloom filters must share m_bits and n_hashes to merge")
        return SparseLoCoBloom(
            bits=(self.bits | other.bits),
            m_bits=self.m_bits,
            n_hashes=self.n_hashes,
        )


def sparse_loco_compress(
    grad: Tensor,
    *,
    density: float,
    peer_bloom: SparseLoCoBloom | None = None,
    bloom_false_pos_rate: float = 0.01,
) -> tuple[CompressedGradient, SparseLoCoBloom]:
    """SparseLoCo (Templar / DIST-05): top-k + Bloom-filter dedup.

    The producing peer:

    1. Takes top-k indices of its local pseudo-gradient.
    2. Drops indices whose Bloom-filter membership says "another peer
       already transmitted this" (when ``peer_bloom`` is given).
    3. Records the kept indices into a fresh Bloom filter for the
       next round of peer-to-peer merge.

    This deduplication achieves the headline 97% compression at
    no accuracy loss because each gradient coordinate gets transmitted
    by exactly one peer per round (in expectation), instead of every
    peer sending the same top-k indices.

    Args:
        grad:                 local pseudo-gradient.
        density:              target top-k density (0, 1].
        peer_bloom:           filter from prior peers' transmissions.
            ``None`` = first peer in the round.
        bloom_false_pos_rate: target false-positive rate for the
            outgoing bloom filter (default 1%).

    Returns:
        ``(compressed, outgoing_bloom)``. The outgoing bloom filter
        contains the indices this peer just sent — it merges with
        peer blooms before being passed to the next peer.
    """
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1]; got {density}")

    n_total = grad.numel()
    k = max(1, round(density * n_total))

    flat = grad.flatten()
    abs_vals = flat.abs()
    # We need more than k initially since some will be filtered by the bloom.
    overshoot = min(n_total, max(2 * k, 64))
    _, candidate_idx = abs_vals.topk(overshoot, sorted=True)

    # Bloom filter for this round's outgoing.
    m_bits, n_hashes = _bloom_size_for(k, false_pos_rate=bloom_false_pos_rate)
    outgoing = SparseLoCoBloom.empty(m_bits=m_bits, n_hashes=n_hashes)

    kept_idx: list[int] = []
    for c in candidate_idx.tolist():
        if peer_bloom is not None and peer_bloom.contains(c):
            continue
        kept_idx.append(c)
        outgoing.add(c)
        if len(kept_idx) >= k:
            break

    if not kept_idx:
        # Fallback — no indices passed the filter; send the single
        # largest-magnitude one to guarantee progress.
        kept_idx = [int(candidate_idx[0].item())]
        outgoing.add(kept_idx[0])

    indices_t = torch.tensor(kept_idx, dtype=torch.long, device=grad.device)
    values_t = flat[indices_t]
    compressed = CompressedGradient(
        indices=indices_t, values=values_t,
        shape=tuple(grad.shape), n_total=n_total,
    )
    return compressed, outgoing
