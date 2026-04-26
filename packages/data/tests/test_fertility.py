"""Tests for per-language fertility measurement."""

from __future__ import annotations

import pytest
from saint_llm_data import (
    CharTokenizer,
    FertilityRecord,
    FertilityReport,
    measure_fertility,
    measure_per_language_fertility,
)


def _tok() -> CharTokenizer:
    """CharTokenizer is fertility = 1 by construction (1 token / codepoint)."""
    return CharTokenizer(base_vocab=16, unicode_max=0x10000)


def test_measure_fertility_empty_text() -> None:
    chars, tokens = measure_fertility(_tok(), "")
    assert chars == 0 and tokens == 0


def test_measure_fertility_chartokenizer_is_one_per_char() -> None:
    chars, tokens = measure_fertility(_tok(), "hello")
    assert chars == 5
    assert tokens == 5


def test_measure_per_language_basic() -> None:
    samples = {
        "en": ["hello world"],
        "uk": ["привіт"],  # 6 chars Cyrillic
    }
    report = measure_per_language_fertility(_tok(), samples)
    by = report.by_language()
    assert pytest.approx(by["en"].fertility) == 1.0
    assert pytest.approx(by["uk"].fertility) == 1.0
    assert by["en"].chars == 11
    assert by["uk"].chars == 6


def test_records_sorted_alphabetically() -> None:
    samples = {"zh": ["你好"], "uk": ["а"], "en": ["a"]}  # noqa: RUF001
    report = measure_per_language_fertility(_tok(), samples)
    langs = [r.language for r in report.records]
    assert langs == ["en", "uk", "zh"]


def test_zero_char_languages_skipped() -> None:
    samples = {"en": ["hello"], "fr": [], "es": [""]}
    report = measure_per_language_fertility(_tok(), samples)
    by = report.by_language()
    assert "en" in by
    assert "fr" not in by
    assert "es" not in by


def test_to_dict_round_trip() -> None:
    samples = {"en": ["abc"]}
    report = measure_per_language_fertility(_tok(), samples)
    d = report.to_dict()
    assert "records" in d
    assert isinstance(d["records"], list)
    assert d["records"][0]["language"] == "en"
    assert d["records"][0]["chars"] == 3
    assert d["records"][0]["tokens"] == 3
    assert d["records"][0]["fertility"] == 1.0


def test_fertility_record_immutable() -> None:
    rec = FertilityRecord("en", 1, 3, 3, 1.0)
    with pytest.raises((AttributeError, TypeError)):
        rec.fertility = 2.0  # type: ignore[misc]


def test_fertility_report_immutable() -> None:
    rec = FertilityRecord("en", 1, 3, 3, 1.0)
    report = FertilityReport(records=(rec,))
    with pytest.raises((AttributeError, TypeError)):
        report.records = ()  # type: ignore[misc]
