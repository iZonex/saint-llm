"""Tests for v0.0 tokenizer-trainer extensions: SAINT_V0_0_SPECIAL_TOKENS,
SAINT_V0_0_FORCE_INCLUDE_CHARS, force_include_chars argument to train_bbpe.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from saint_llm_data import (
    SAINT_V0_0_FORCE_INCLUDE_CHARS,
    SAINT_V0_0_SPECIAL_TOKENS,
    train_bbpe,
)


@pytest.fixture
def small_corpus() -> list[str]:
    """Small UK + EN mixed corpus suitable for fast tests."""
    return [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test",
        "привіт світ це тест",
        "ключі мрії дощ ґанок",
        "the apple's color is red and the dog's tail wags",
        "she's running and they've finished the work",
        "ключі ґанок мрії птах",
        # repeat several times to give the trainer more data
    ] * 50


def test_saint_v0_0_special_tokens_includes_v0_0_picks(tmp_path: Path) -> None:
    """v0.0 special tokens must include effort tiers (RL-07/ADR-0017),
    memory verbs (MEM-01/ADR-0008), and core control tokens.
    """
    expected_subset = {
        "<pad>",
        "<|endoftext|>",
        "<|effort:0|>",
        "<|effort:1|>",
        "<|effort:2|>",
        "<|effort:3|>",
        "<|effort:4|>",
        "<|memory_recall|>",
        "<|memory_save|>",
        "<|memory_result|>",
        "<|reflect|>",
        "</reflect|>",
        "<|image_pad|>",
        "<|audio_start|>",
    }
    assert expected_subset.issubset(set(SAINT_V0_0_SPECIAL_TOKENS))


def test_saint_v0_0_force_include_chars_includes_uk_letters() -> None:
    """TOK-03 mandates UK Cyrillic letters + apostrophe variants."""
    expected_subset = {"ї", "Ї", "є", "Є", "ґ", "Ґ", "'", "ʼ", "’"}  # noqa: RUF001
    assert expected_subset.issubset(set(SAINT_V0_0_FORCE_INCLUDE_CHARS))


def test_train_bbpe_with_force_include_chars(tmp_path: Path, small_corpus: list[str]) -> None:
    """Training with force_include_chars produces a tokenizer that has the
    forced characters available — they encode as one token at minimum
    (covered by initial alphabet so guaranteed).
    """
    output = tmp_path / "tokenizer.json"
    tok = train_bbpe(
        small_corpus,
        output,
        vocab_size=500,  # tiny for fast test
        min_frequency=1,
        special_tokens=SAINT_V0_0_SPECIAL_TOKENS,
        force_include_chars=SAINT_V0_0_FORCE_INCLUDE_CHARS,
        show_progress=False,
    )
    assert output.exists()
    # Encode a UK string that uses all forced chars; should produce a
    # bounded-length token sequence (not unkkified).
    ids = tok.encode("ї Ї є Є ґ Ґ ' ʼ ’")  # noqa: RUF001
    assert len(ids) > 0
    decoded = tok.decode(ids)
    # Decoder roundtrip must preserve all forced characters.
    for ch in "їЇєЄґҐ":
        assert ch in decoded


def test_train_bbpe_v0_0_special_tokens_at_low_ids(tmp_path: Path, small_corpus: list[str]) -> None:
    """Special tokens occupy the lowest IDs in encoding order."""
    output = tmp_path / "tokenizer.json"
    tok = train_bbpe(
        small_corpus,
        output,
        vocab_size=500,
        min_frequency=1,
        special_tokens=SAINT_V0_0_SPECIAL_TOKENS,
        show_progress=False,
    )
    # All v0.0 special tokens should resolve to ids in [0, len(specials)) range
    # since BpeTrainer reserves them at the front.
    n_specials = len(SAINT_V0_0_SPECIAL_TOKENS)
    backend = tok._backend  # type: ignore[attr-defined]
    for special in SAINT_V0_0_SPECIAL_TOKENS:
        tid = backend.token_to_id(special)
        assert tid is not None
        assert 0 <= tid < n_specials


def test_train_bbpe_default_force_include_empty_does_not_break(
    tmp_path: Path, small_corpus: list[str]) -> None:
    """force_include_chars defaults to empty tuple; train_bbpe still works."""
    output = tmp_path / "tokenizer.json"
    tok = train_bbpe(
        small_corpus,
        output,
        vocab_size=500,
        min_frequency=1,
        show_progress=False,
    )
    assert output.exists()
    ids = tok.encode("hello world")
    assert len(ids) > 0
