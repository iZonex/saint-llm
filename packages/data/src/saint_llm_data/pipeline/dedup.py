"""Deduplication stages.

v0.0 ships ``FingerprintDedup`` — exact-document SHA1 dedup with an
in-memory seen-set. Correct for exact duplicates, scales to millions
of documents, doesn't catch near-duplicates or paraphrases.

LSHBloom (DATA-04 / arxiv 2411.04257) is the production v0.1 upgrade:
Bloom-filter approximation of MinHash-LSH, 12x faster than full
MinHash-LSH at internet scale. Defer the full LSHBloom implementation
until the v0.0 fingerprint dedup is validated in real runs.

This module also exposes ``minhash_signature(text, n_perm)`` so
v0.1 LSHBloom can be a drop-in replacement reading the same
shingled-document interface.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from saint_llm_data.pipeline.stage import Document


def _shingles(text: str, n: int = 5) -> set[str]:
    """Character n-grams used for MinHash. Default n=5."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def minhash_signature(text: str, *, n_perm: int = 128) -> tuple[int, ...]:
    """Compute an n-permutation MinHash signature over 5-shingles.

    Pure-Python; no external deps. Suitable for ~1M-doc-scale dedup
    research and for dropping into LSHBloom as the per-doc summary.
    """
    shing = _shingles(text)
    if not shing:
        return tuple(0 for _ in range(n_perm))
    sig: list[int] = []
    for i in range(n_perm):
        # Salted hash per permutation. Python hash() is process-stable
        # in tests because we set PYTHONHASHSEED in CI; for reproducible
        # signatures across processes use sha1 instead.
        salt = str(i).encode()
        min_h = min(
            int.from_bytes(
                hashlib.sha1(salt + s.encode()).digest()[:8],
                byteorder="big",
                signed=False,
            )
            for s in shing
        )
        sig.append(min_h)
    return tuple(sig)


@dataclass
class FingerprintDedup:
    """Exact-document dedup via SHA1 fingerprint set.

    Drops every document whose ``text`` hash has been seen before.
    State is in-memory; instantiate one per pipeline run, or share
    across slices for cross-language dedup (set ``cross_language=True``
    in the orchestrator).
    """

    name: str = "fingerprint_dedup"
    _seen: set[str] = field(default_factory=set)

    def __call__(self, doc: Document) -> Document | None:
        text = doc.get("text", "")
        if not text:
            return doc
        fp = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if fp in self._seen:
            return None
        self._seen.add(fp)
        return doc

    def __len__(self) -> int:
        return len(self._seen)

    def reset(self) -> None:
        self._seen.clear()


@dataclass
class MinHashDedup:
    """MinHash-based near-duplicate dedup (signature exact match).

    Two documents with identical MinHash signatures are treated as
    duplicates. This is a strict version of MinHash-LSH (full LSH
    bands the signature into b chunks; here we treat the full signature
    as the deduplication key). Useful for catching paraphrased
    duplicates that exact fingerprinting misses.

    Production scale (>>1M docs) should use LSHBloom (v0.1).
    """

    n_perm: int = 128
    name: str = "minhash_dedup"
    _seen: set[tuple[int, ...]] = field(default_factory=set)

    def __call__(self, doc: Document) -> Document | None:
        text = doc.get("text", "")
        if not text:
            return doc
        sig = minhash_signature(text, n_perm=self.n_perm)
        if sig in self._seen:
            return None
        self._seen.add(sig)
        return doc

    def __len__(self) -> int:
        return len(self._seen)

    def reset(self) -> None:
        self._seen.clear()
