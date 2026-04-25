"""HF-datasets wrapper: in-memory dataset path covers the wiring without network."""

from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset, load_from_disk
from saint_llm_data import CharTokenizer, HuggingFaceTextDataset


def _save_local_dataset(tmp_path: Path, rows: list[dict[str, str]]) -> str:
    """Persist an HF Dataset to disk so load_dataset can pick it up."""
    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(tmp_path / "ds"))
    return str(tmp_path / "ds")


def test_hf_dataset_streams_packed_batches(tmp_path: Path) -> None:
    """Build a tiny on-disk HF dataset and stream it through the wrapper."""
    rows = [{"text": "hello"}, {"text": "world"}, {"text": "foo bar"}]
    path = _save_local_dataset(tmp_path, rows)
    # We use load_from_disk rather than load_dataset, so reach in via
    # load_dataset_kwargs path. Easier: monkey-patch the _open method.
    tok = CharTokenizer()
    ds = HuggingFaceTextDataset(
        path="placeholder",
        tokenizer=tok,
        seq_len=8,
        batch_size=1,
        streaming=False,
        drop_last=False,
    )
    # Override _open to bypass load_dataset.
    loaded = load_from_disk(path)
    ds._open = lambda: loaded  # type: ignore[method-assign]

    batches = list(ds)
    assert len(batches) >= 1
    flat: list[int] = []
    for b in batches:
        flat.extend(b.tokens.flatten().tolist())
    # Three docs → three EOS tokens.
    assert flat.count(tok.eos_token_id) == 3


def test_hf_dataset_missing_text_field_raises(tmp_path: Path) -> None:
    rows = [{"body": "hello"}]
    path = _save_local_dataset(tmp_path, rows)
    tok = CharTokenizer()
    ds = HuggingFaceTextDataset(
        path="placeholder",
        tokenizer=tok,
        seq_len=8,
        text_field="text",  # but rows have "body"
        streaming=False,
        drop_last=False,
    )
    loaded = load_from_disk(path)
    ds._open = lambda: loaded  # type: ignore[method-assign]

    with pytest.raises(KeyError, match="missing field 'text'"):
        list(ds)


def test_hf_dataset_skips_empty_rows(tmp_path: Path) -> None:
    rows = [{"text": "hello"}, {"text": ""}, {"text": "world"}, {"text": ""}]
    path = _save_local_dataset(tmp_path, rows)
    tok = CharTokenizer()
    ds = HuggingFaceTextDataset(
        path="placeholder", tokenizer=tok, seq_len=8, streaming=False, drop_last=False,
    )
    loaded = load_from_disk(path)
    ds._open = lambda: loaded  # type: ignore[method-assign]

    flat: list[int] = []
    for b in ds:
        flat.extend(b.tokens.flatten().tolist())
    assert flat.count(tok.eos_token_id) == 2


def test_hf_dataset_custom_text_field(tmp_path: Path) -> None:
    rows = [{"body": "alpha"}, {"body": "beta"}]
    path = _save_local_dataset(tmp_path, rows)
    tok = CharTokenizer()
    ds = HuggingFaceTextDataset(
        path="placeholder", tokenizer=tok, seq_len=8,
        text_field="body", streaming=False, drop_last=False,
    )
    loaded = load_from_disk(path)
    ds._open = lambda: loaded  # type: ignore[method-assign]

    flat: list[int] = []
    for b in ds:
        flat.extend(b.tokens.flatten().tolist())
    assert flat.count(tok.eos_token_id) == 2


def test_hf_dataset_non_string_field_raises(tmp_path: Path) -> None:
    """text_field pointing at a non-string column must error clearly."""
    rows = [{"text": 1}, {"text": 2}]  # uniform int column
    path = _save_local_dataset(tmp_path, rows)
    tok = CharTokenizer()
    ds = HuggingFaceTextDataset(
        path="placeholder", tokenizer=tok, seq_len=8, streaming=False, drop_last=False,
    )
    loaded = load_from_disk(path)
    ds._open = lambda: loaded  # type: ignore[method-assign]

    with pytest.raises(TypeError, match="not a string"):
        list(ds)
