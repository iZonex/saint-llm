"""Multi-source proportional corpus iterator.

For tokenizer training (D0.0.1) and downstream pretraining we need to
draw documents from N text sources with controlled per-source proportions
— per ADR-0018 (corpus mix) and ADR-0020 (RegMix mixture optimization).

The iterator is a *streaming* interleaver: it never materializes all
slices; each slice is itself a streaming iterable (HF dataset stream,
JSONL file reader, etc.). On every yield it picks one slice according
to the configured weights and emits one document from that slice.

Slice exhaustion: if a slice runs out, it is removed from the active
pool; remaining weights are renormalized. This means the tail of the
mixed stream is whichever slice has the most documents. Caller controls
the total via ``max_docs`` (cap regardless of weights).

Reproducibility: the slice selection is deterministic given a seed;
within a slice the underlying HF dataset's order applies.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusSlice:
    """One named source in a mixed-corpus draw.

    Attributes:
        name:    short identifier for logging (e.g. ``"hplt-en"``).
        weight:  unnormalized sampling weight (positive). Weights are
            normalized across active slices at construction and re-
            normalized as slices exhaust.
        source:  iterable producing strings (one document per element).
            Must be re-iterable only if the caller needs multiple passes;
            ``MixedCorpus`` consumes a single pass.
    """

    name: str
    weight: float
    source: Iterable[str]


class MixedCorpus:
    """Draw documents from multiple sources in configured proportion.

    Construct from a list of :class:`CorpusSlice`. Iterate to receive
    documents. Stops when ``max_docs`` is reached or all slices are
    exhausted, whichever comes first.

    The iterator emits ``str`` (raw document text) so it composes
    directly with :func:`saint_llm_data.tokenizer_trainer.train_bbpe`
    and similar consumers.
    """

    def __init__(
        self,
        slices: list[CorpusSlice],
        *,
        max_docs: int | None = None,
        seed: int = 0,
    ) -> None:
        if not slices:
            raise ValueError("MixedCorpus requires at least one slice")
        for s in slices:
            if s.weight <= 0:
                raise ValueError(f"slice {s.name!r} has non-positive weight {s.weight}")
        self._slices = slices
        self._max_docs = max_docs
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[str]:
        # Materialize per-slice iterators once.
        active: list[tuple[str, float, Iterator[str]]] = [
            (s.name, s.weight, iter(s.source)) for s in self._slices
        ]
        emitted = 0
        while active and (self._max_docs is None or emitted < self._max_docs):
            names = [a[0] for a in active]
            weights = [a[1] for a in active]
            iters = [a[2] for a in active]
            idx = self._weighted_pick(weights)
            try:
                doc = next(iters[idx])
            except StopIteration:
                # Drop exhausted slice and renormalize implicitly via removal.
                _ = names.pop(idx)
                weights.pop(idx)
                iters.pop(idx)
                active = list(zip(names, weights, iters, strict=True))
                continue
            yield doc
            emitted += 1

    def _weighted_pick(self, weights: list[float]) -> int:
        """Return an index into ``weights`` proportional to its values."""
        total = sum(weights)
        r = self._rng.uniform(0.0, total)
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if r <= upto:
                return i
        return len(weights) - 1
