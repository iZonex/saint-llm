"""ParquetShardDataset — stream pre-tokenized parquet shards into the trainer.

The v0.0 data pipeline (``experiments/build_pretrain_data.py``) writes
sharded parquet output where each row is one tokenized document with
columns:

* ``token_ids: list[int64]`` — output of TokenizeStage
* ``language: str`` — ISO 639-3 code
* ``slice: str`` — source slice name

This dataset reads those shards in order, packs token streams into
fixed-length windows via ``pack_into_batch``, and yields ``PackedBatch``
items ready for the Trainer.

Subclasses ``torch.utils.data.IterableDataset`` with shard-level worker
sharding so multiple DataLoader workers process disjoint shards.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from torch.utils.data import IterableDataset, get_worker_info

from saint_llm_data.packing import PackedBatch, pack_into_batch
from saint_llm_data.tokenizer import Tokenizer


class ParquetShardDataset(IterableDataset):
    """Stream tokenized documents from sharded parquet output.

    Args:
        shard_dir:    directory containing ``*.parquet`` files written by
            ``experiments/build_pretrain_data.py``.
        tokenizer:    used only for ``eos_token_id`` and ``pad_token_id``
            during packing — token IDs themselves are read from the
            parquet ``token_ids`` column.
        seq_len:      fixed-window length for packing.
        batch_size:   yield ``(batch_size, seq_len)`` batches.
        drop_last:    drop residual partial batch at end-of-shard.
        shard_glob:   glob pattern for shard discovery (default ``*.parquet``).
    """

    def __init__(
        self,
        shard_dir: str | Path,
        *,
        tokenizer: Tokenizer,
        seq_len: int,
        batch_size: int = 1,
        drop_last: bool = True,
        shard_glob: str = "*.parquet",
    ) -> None:
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shard_glob = shard_glob
        if not self.shard_dir.exists():
            raise FileNotFoundError(f"shard_dir not found: {self.shard_dir}")

    def _list_shards(self) -> list[Path]:
        return sorted(self.shard_dir.glob(self.shard_glob))

    def _shards_for_worker(self) -> list[Path]:
        """Return the shards this worker is responsible for."""
        shards = self._list_shards()
        info = get_worker_info()
        if info is None:
            return shards
        return [s for i, s in enumerate(shards) if i % info.num_workers == info.id]

    def _stream_token_ids(self, shards: Iterable[Path]) -> Iterator[list[int]]:
        # Local import — pyarrow is already a `data` package dep, but
        # importing it here keeps module-import side-effect-free for
        # users who only need other parts of saint_llm_data.
        import pyarrow.parquet as pq  # noqa: PLC0415

        for shard in shards:
            table = pq.read_table(str(shard), columns=["token_ids"])
            col = table.column("token_ids").to_pylist()
            for ids in col:
                if ids:
                    yield list(ids)

    def __iter__(self) -> Iterator[PackedBatch]:
        shards = self._shards_for_worker()
        token_streams = self._stream_token_ids(shards)
        yield from pack_into_batch(
            token_streams,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            drop_last=self.drop_last,
        )
