"""TextFileDataset: plain text + JSONL paths, eos placement, batch shape."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from saint_llm_data import CharTokenizer, TextFileDataset


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_text_file_dataset_basic_streaming(tmp_path: Path) -> None:
    file_path = _write(tmp_path / "corpus.txt", ["hello", "world", "foo bar baz"])
    tok = CharTokenizer()
    ds = TextFileDataset(file_path, tokenizer=tok, seq_len=8, batch_size=1, drop_last=False)
    batches = list(ds)
    assert len(batches) >= 1
    flat: list[int] = []
    for b in batches:
        flat.extend(b.tokens.flatten().tolist())
    # Three documents → three EOS tokens in the stream.
    assert flat.count(tok.eos_token_id) == 3


def test_text_file_dataset_skips_blank_lines(tmp_path: Path) -> None:
    file_path = _write(tmp_path / "blanks.txt", ["", "hello", "", "world", ""])
    tok = CharTokenizer()
    ds = TextFileDataset(file_path, tokenizer=tok, seq_len=8, drop_last=False)
    flat: list[int] = []
    for b in ds:
        flat.extend(b.tokens.flatten().tolist())
    assert flat.count(tok.eos_token_id) == 2


def test_text_file_dataset_jsonl_mode(tmp_path: Path) -> None:
    file_path = tmp_path / "corpus.jsonl"
    payloads = [{"text": "hello"}, {"text": "world"}, {"other": "ignored"}, {"text": "ok"}]
    file_path.write_text(
        "\n".join(json.dumps(p) for p in payloads) + "\n",
        encoding="utf-8",
    )
    tok = CharTokenizer()
    # Skipping the row with missing 'text' would require pre-filtering — instead
    # we expect a clear KeyError when consumed.
    ds = TextFileDataset(file_path, tokenizer=tok, seq_len=8, jsonl=True, drop_last=False)
    with pytest.raises(KeyError, match="missing field"):
        list(ds)


def test_text_file_dataset_jsonl_happy_path(tmp_path: Path) -> None:
    file_path = tmp_path / "corpus.jsonl"
    payloads = [{"text": "hello"}, {"text": "world"}]
    file_path.write_text(
        "\n".join(json.dumps(p) for p in payloads) + "\n",
        encoding="utf-8",
    )
    tok = CharTokenizer()
    ds = TextFileDataset(file_path, tokenizer=tok, seq_len=8, jsonl=True, drop_last=False)
    batches = list(ds)
    flat: list[int] = []
    for b in batches:
        flat.extend(b.tokens.flatten().tolist())
    assert flat.count(tok.eos_token_id) == 2


def test_text_file_dataset_jsonl_custom_field(tmp_path: Path) -> None:
    file_path = tmp_path / "corpus.jsonl"
    payloads = [{"body": "hello"}, {"body": "world"}]
    file_path.write_text(
        "\n".join(json.dumps(p) for p in payloads) + "\n",
        encoding="utf-8",
    )
    tok = CharTokenizer()
    ds = TextFileDataset(
        file_path, tokenizer=tok, seq_len=8, jsonl=True, json_field="body", drop_last=False,
    )
    flat: list[int] = []
    for b in ds:
        flat.extend(b.tokens.flatten().tolist())
    assert flat.count(tok.eos_token_id) == 2


def test_text_file_dataset_batch_size_groups_rows(tmp_path: Path) -> None:
    """Many short docs → multiple windows → grouped into batch_size=2 batches."""
    docs = ["abcdefgh"] * 12
    file_path = _write(tmp_path / "many.txt", docs)
    tok = CharTokenizer()
    ds = TextFileDataset(
        file_path, tokenizer=tok, seq_len=8, batch_size=2, drop_last=True,
    )
    batches = list(ds)
    assert all(b.tokens.shape == (2, 8) for b in batches)


def test_text_file_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TextFileDataset(
            tmp_path / "nope.txt", tokenizer=CharTokenizer(), seq_len=8,
        )


def test_text_file_dataset_iterates_fresh_each_loop(tmp_path: Path) -> None:
    """__iter__ should produce a new generator each call (single-shot semantics)."""
    file_path = _write(tmp_path / "twice.txt", ["a", "b"])
    tok = CharTokenizer()
    ds = TextFileDataset(file_path, tokenizer=tok, seq_len=8, drop_last=False)
    first = list(ds)
    second = list(ds)
    # Both passes should yield the same content.
    assert len(first) == len(second)
    assert (first[0].tokens == second[0].tokens).all()
