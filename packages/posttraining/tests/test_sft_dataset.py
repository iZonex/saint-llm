"""Tests for JsonlSFTDataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from saint_llm_data import CharTokenizer
from saint_llm_posttraining import JsonlSFTDataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_jsonl_sft_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        JsonlSFTDataset(
            tmp_path / "nope.jsonl",
            tokenizer=CharTokenizer(),
            seq_len=8,
        )


def test_jsonl_sft_dataset_yields_packed_batches(tmp_path: Path) -> None:
    p = tmp_path / "sft.jsonl"
    _write_jsonl(p, [
        {"prompt": "hi", "response": "hello"},
        {"prompt": "what?", "response": "this"},
    ])
    ds = JsonlSFTDataset(
        p, tokenizer=CharTokenizer(), seq_len=12, drop_last=False,
    )
    batches = list(ds)
    assert len(batches) >= 1
    for b in batches:
        assert b.tokens.shape == (1, 12)
        assert b.loss_mask.shape == (1, 12)
        assert b.segment_ids.shape == (1, 12)


def test_jsonl_sft_dataset_skips_invalid_lines(tmp_path: Path) -> None:
    p = tmp_path / "sft.jsonl"
    p.write_text(
        "not json at all\n"
        + json.dumps({"prompt": "ok", "response": "yes"}) + "\n"
        + json.dumps({"prompt": 42, "response": "wrong-type"}) + "\n"
        + json.dumps({"missing_response": "x"}) + "\n",
        encoding="utf-8",
    )
    ds = JsonlSFTDataset(
        p, tokenizer=CharTokenizer(), seq_len=8, drop_last=False,
    )
    # Just one valid example; should yield at least 1 batch (or 0 if too short).
    batches = list(ds)
    # At minimum no exception.
    assert all(b.tokens.shape == (1, 8) for b in batches)


def test_jsonl_sft_dataset_custom_field_names(tmp_path: Path) -> None:
    p = tmp_path / "sft.jsonl"
    _write_jsonl(p, [
        {"q": "hi", "a": "hello world"},
    ])
    ds = JsonlSFTDataset(
        p, tokenizer=CharTokenizer(), seq_len=20,
        prompt_field="q", response_field="a",
        system_field=None, drop_last=False,
    )
    batches = list(ds)
    assert len(batches) >= 1


def test_jsonl_sft_dataset_system_prefix_loss_masked(tmp_path: Path) -> None:
    p = tmp_path / "sft.jsonl"
    _write_jsonl(p, [
        {"system": "You are helpful.", "prompt": "Hi", "response": "Hello."},
    ])
    ds = JsonlSFTDataset(
        p, tokenizer=CharTokenizer(), seq_len=64,
        drop_last=False,
    )
    batches = list(ds)
    assert len(batches) >= 1
    # First non-zero mask position is after the system + prompt section.
    mask_row = batches[0].loss_mask[0].tolist()
    # system "You are helpful." + "\n" = 17 chars; prompt "Hi" = 2 chars.
    # First 19 positions should be loss-masked (0).
    assert all(m == 0 for m in mask_row[:19])
    # Response positions should be unmasked (1).
    assert any(m == 1 for m in mask_row[19:])
