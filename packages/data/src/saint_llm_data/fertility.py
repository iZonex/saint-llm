"""Per-language tokenizer fertility measurement.

Fertility = average number of tokens per character. Lower is better
(closer to 1.0 means the tokenizer represents the language efficiently).
TOK-01 in AUGMENTATIONS targets UK fertility 1.65-1.75; ADR-0021 makes
this a measurement gate for whether MorphBPE evaluation is required.

This module provides:

* :func:`measure_fertility(tokenizer, text)` — single-text fertility.
* :func:`measure_per_language_fertility(tokenizer, samples)` — bulk
  measurement across language-keyed text samples; returns a
  :class:`FertilityReport`.

The report serializes to JSON for committal to
``runs/v0.0/tokenizer-fertility.json`` (per ADR-0021 D0.0.1 exit
checklist).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from saint_llm_data.tokenizer import Tokenizer


@dataclass(frozen=True)
class FertilityRecord:
    """One language's fertility measurement.

    Attributes:
        language:    short language code (``"en"``, ``"uk"``, etc.).
        n_samples:   number of text samples measured.
        chars:       total character count across samples.
        tokens:      total token count produced by the tokenizer.
        fertility:   ``tokens / chars``. Lower is better.
    """

    language: str
    n_samples: int
    chars: int
    tokens: int
    fertility: float


@dataclass(frozen=True)
class FertilityReport:
    """Per-language fertility report.

    Use ``to_dict()`` for JSON serialization.
    """

    records: tuple[FertilityRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {"records": [asdict(r) for r in self.records]}

    def by_language(self) -> dict[str, FertilityRecord]:
        return {r.language: r for r in self.records}


def measure_fertility(tokenizer: Tokenizer, text: str) -> tuple[int, int]:
    """Return ``(chars, tokens)`` for a single text.

    Counts characters with ``len(text)`` (Python str length, codepoints).
    Counts tokens by encoding via ``tokenizer.encode`` and taking the
    length of the returned id list.
    """
    if not text:
        return 0, 0
    chars = len(text)
    tokens = len(tokenizer.encode(text))
    return chars, tokens


def measure_per_language_fertility(
    tokenizer: Tokenizer,
    samples: Mapping[str, Iterable[str]],
) -> FertilityReport:
    """Compute fertility across language-keyed text samples.

    Args:
        tokenizer: any object satisfying ``Tokenizer`` Protocol.
        samples: mapping ``language_code -> iterable[str]`` of evaluation
            texts. Iterables are consumed once.

    Returns:
        ``FertilityReport`` with one record per language. Languages with
        zero characters in their samples are skipped.
    """
    records: list[FertilityRecord] = []
    for lang, texts in samples.items():
        n_samples = 0
        total_chars = 0
        total_tokens = 0
        for text in texts:
            n_samples += 1
            chars, tokens = measure_fertility(tokenizer, text)
            total_chars += chars
            total_tokens += tokens
        if total_chars == 0:
            continue
        records.append(
            FertilityRecord(
                language=lang,
                n_samples=n_samples,
                chars=total_chars,
                tokens=total_tokens,
                fertility=total_tokens / total_chars,
            ),
        )
    # Sort alphabetically for stable output across runs.
    records.sort(key=lambda r: r.language)
    return FertilityReport(records=tuple(records))
