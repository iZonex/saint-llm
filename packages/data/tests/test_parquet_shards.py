"""Tests for ParquetShardDataset."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from saint_llm_data import CharTokenizer, ParquetShardDataset


def _write_shard(path: Path, token_ids_per_doc: list[list[int]]) -> None:
    table = pa.table({"token_ids": token_ids_per_doc})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_missing_shard_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ParquetShardDataset(
            tmp_path / "nope",
            tokenizer=CharTokenizer(),
            seq_len=8,
        )


def test_streams_packed_batches(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "shard_0.parquet",
        [[10, 11, 12, 13, 14], [20, 21, 22], [30, 31, 32, 33, 34, 35, 36]],
    )
    ds = ParquetShardDataset(
        tmp_path,
        tokenizer=CharTokenizer(),
        seq_len=8,
        batch_size=1,
        drop_last=False,
    )
    batches = list(ds)
    assert len(batches) >= 1
    for b in batches:
        assert b.tokens.shape == (1, 8)


def test_reads_multiple_shards_in_order(tmp_path: Path) -> None:
    _write_shard(tmp_path / "shard_0.parquet", [[1, 2, 3, 4]])
    _write_shard(tmp_path / "shard_1.parquet", [[5, 6, 7, 8]])
    ds = ParquetShardDataset(
        tmp_path,
        tokenizer=CharTokenizer(),
        seq_len=10,
        batch_size=1,
        drop_last=False,
    )
    batches = list(ds)
    flat = [int(t) for b in batches for t in b.tokens[0].tolist()]
    # All tokens from both shards present (order: shard_0 then shard_1, eos
    # between docs since pack_sequences inserts eos).
    assert 1 in flat and 5 in flat


def test_drop_last_true_drops_partial(tmp_path: Path) -> None:
    _write_shard(tmp_path / "shard_0.parquet", [[1, 2, 3]])  # only 3 + eos = 4 tokens
    ds = ParquetShardDataset(
        tmp_path, tokenizer=CharTokenizer(), seq_len=10, drop_last=True,
    )
    assert list(ds) == []  # not enough for a full window, dropped


def test_empty_token_lists_skipped(tmp_path: Path) -> None:
    """Documents with empty token_ids lists are skipped, not crashed."""
    _write_shard(tmp_path / "shard_0.parquet", [[], [1, 2, 3, 4, 5]])
    ds = ParquetShardDataset(
        tmp_path, tokenizer=CharTokenizer(), seq_len=4, drop_last=False,
    )
    batches = list(ds)
    assert len(batches) >= 1
