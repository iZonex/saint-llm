"""Quality filter stages.

Per ADR-0019 the v0.0 production quality classifier is **Ultra-FineWeb
fastText** (256-d, n=3, 3 epochs). This module exposes:

* ``QualityClassifier`` — Protocol any quality scorer must satisfy.
* ``LengthQualityFilter`` — minimal stub: drops docs shorter than a
  per-language threshold. No external deps; v0.0 testing fallback
  before fastText models are available.
* ``QualityFilter`` — pipeline stage wrapping any QualityClassifier.

Production deployment plugs in ``FasttextQualityClassifier(model_path,
threshold)`` (a thin wrapper over the ``fasttext`` library) — that
class is part of the runtime image, not committed here, because the
``.bin`` model itself is large and the load path is environment-
dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from saint_llm_data.pipeline.stage import Document


class QualityClassifier(Protocol):
    """Score a document; higher score = higher quality.

    Returns a float in [0, 1] by convention, but stages compare against
    the threshold directly so any monotone scoring works.
    """

    def __call__(self, text: str, *, language: str | None = None) -> float: ...


@dataclass
class QualityFilter:
    """Drop documents below ``threshold`` according to ``classifier``.

    The classifier is called once per document; ``apply_to_slices``
    restricts the filter to specified slice names (e.g. skip pre-
    filtered Nemotron-CC v2 data per ADR-0019). Empty ``apply_to_slices``
    means "apply to every slice".
    """

    classifier: QualityClassifier
    threshold: float
    apply_to_slices: tuple[str, ...] = ()
    name: str = "quality_filter"

    def __call__(self, doc: Document) -> Document | None:
        slice_name = doc.get("slice", "")
        if self.apply_to_slices and slice_name not in self.apply_to_slices:
            return doc  # not in the filter's scope
        text = doc.get("text", "")
        if not text:
            return None
        score = self.classifier(text, language=doc.get("language"))
        if score < self.threshold:
            return None
        return doc


@dataclass
class LengthQualityFilter:
    """Stub quality filter: drops documents shorter than ``min_chars``.

    Per-language thresholds via ``min_chars_per_language`` override the
    default. Used as a fallback before Ultra-FineWeb fastText models
    are available; never the production classifier.
    """

    min_chars: int = 100
    min_chars_per_language: dict[str, int] | None = None
    name: str = "length_quality_filter"

    def __call__(self, doc: Document) -> Document | None:
        text = doc.get("text", "")
        threshold = self.min_chars
        if self.min_chars_per_language is not None:
            lang = doc.get("language", "")
            threshold = self.min_chars_per_language.get(lang, self.min_chars)
        if len(text) < threshold:
            return None
        return doc
