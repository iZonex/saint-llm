"""Tests for MixedCorpus multi-source proportional iterator."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from saint_llm_data import CorpusSlice, MixedCorpus


def _slice(name: str, weight: float, items: list[str]) -> CorpusSlice:
    """Wrap a list as a one-pass iterable so it gets exhausted naturally."""

    def gen() -> Iterator[str]:
        yield from items

    return CorpusSlice(name=name, weight=weight, source=gen())


def test_empty_slices_rejected() -> None:
    with pytest.raises(ValueError, match="at least one slice"):
        MixedCorpus([])


def test_non_positive_weight_rejected() -> None:
    with pytest.raises(ValueError, match="non-positive weight"):
        MixedCorpus([CorpusSlice(name="x", weight=0.0, source=iter(["doc"]))])


def test_yields_at_most_max_docs() -> None:
    mix = MixedCorpus(
        [_slice("a", 1.0, ["a"] * 100), _slice("b", 1.0, ["b"] * 100)],
        max_docs=10,
        seed=0,
    )
    docs = list(mix)
    assert len(docs) == 10


def test_terminates_when_all_slices_exhaust() -> None:
    mix = MixedCorpus(
        [_slice("a", 1.0, ["a"] * 3), _slice("b", 1.0, ["b"] * 3)],
        seed=0,
    )
    docs = list(mix)
    # No max_docs cap, all slices fully drained.
    assert len(docs) == 6
    assert docs.count("a") == 3
    assert docs.count("b") == 3


def test_proportions_approximately_match_weights() -> None:
    mix = MixedCorpus(
        [
            _slice("heavy", 9.0, ["h"] * 10_000),
            _slice("light", 1.0, ["l"] * 10_000),
        ],
        max_docs=10_000,
        seed=42,
    )
    docs = list(mix)
    h_count = docs.count("h")
    # Expect ~9000 heavy, ~1000 light. Allow 5% tolerance.
    assert 8500 <= h_count <= 9500


def test_continues_after_one_slice_exhausts() -> None:
    """When a slice runs out, drawing continues from the rest."""
    mix = MixedCorpus(
        [
            _slice("small", 1.0, ["s"] * 5),
            _slice("big", 1.0, ["b"] * 1000),
        ],
        max_docs=100,
        seed=1,
    )
    docs = list(mix)
    # All 5 small docs eventually emitted; rest are big.
    assert docs.count("s") == 5
    assert docs.count("b") == 95


def test_seed_is_deterministic() -> None:
    args = ([_slice("a", 1.0, ["a"] * 100), _slice("b", 1.0, ["b"] * 100)],)
    mix1 = MixedCorpus(*args, max_docs=20, seed=7)
    mix2 = MixedCorpus(
        [_slice("a", 1.0, ["a"] * 100), _slice("b", 1.0, ["b"] * 100)],
        max_docs=20,
        seed=7,
    )
    assert list(mix1) == list(mix2)


def test_different_seeds_produce_different_orderings() -> None:
    mk = lambda: [_slice("a", 1.0, ["a"] * 100), _slice("b", 1.0, ["b"] * 100)]  # noqa: E731
    docs1 = list(MixedCorpus(mk(), max_docs=20, seed=0))
    docs2 = list(MixedCorpus(mk(), max_docs=20, seed=1))
    # Same composition but different ordering (probabilistically).
    assert docs1 != docs2
