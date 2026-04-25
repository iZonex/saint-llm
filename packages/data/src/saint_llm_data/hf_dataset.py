"""Stream HuggingFace datasets → tokenizer → packed batches.

Thin wrapper over ``datasets.load_dataset`` that pipes the configured text
field through a ``Tokenizer`` and yields ``PackedBatch`` windows ready for the
training loop. ``streaming=True`` by default — for any non-trivial corpus we
do *not* want to download the whole thing locally.

Subclasses ``torch.utils.data.IterableDataset`` with row-level worker
sharding: under a DataLoader with ``num_workers > 0`` each worker tokenizes
1/N of the source rows (round-robin by row index). For HF's own
``IterableDataset.shard``, prefer ``load_dataset_kwargs`` to handle
that upstream where supported.

For local files use ``TextFileDataset`` instead — same output contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from datasets import load_dataset
from torch.utils.data import IterableDataset

from saint_llm_data.dataset import _worker_shard
from saint_llm_data.packing import PackedBatch, pack_into_batch
from saint_llm_data.tokenizer import Tokenizer


class HuggingFaceTextDataset(IterableDataset):
    """Iterable HF-Hub-backed dataset producing ``PackedBatch`` windows.

    Args:
        path: dataset name or path. Forwarded to ``datasets.load_dataset``.
        tokenizer: anything implementing the ``Tokenizer`` Protocol.
        seq_len: target window length.
        batch_size: number of rows per emitted batch.
        split: HF split selector (e.g., ``"train[:1000]"``).
        text_field: name of the column containing the document text.
        streaming: if True, ``load_dataset(..., streaming=True)`` so rows are
            pulled on demand. Disable for tiny / pre-downloaded datasets that
            you'd rather have fully resident.
        config: optional HF "name" / "config" arg for multi-config datasets.
        drop_last: forwarded to ``pack_into_batch``.
        load_dataset_kwargs: passthrough escape hatch for any other
            ``load_dataset`` kwargs (e.g., ``token=...``, ``trust_remote_code=...``).
    """

    def __init__(
        self,
        path: str,
        tokenizer: Tokenizer,
        *,
        seq_len: int,
        batch_size: int = 1,
        split: str = "train",
        text_field: str = "text",
        streaming: bool = True,
        config: str | None = None,
        drop_last: bool = True,
        load_dataset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.split = split
        self.text_field = text_field
        self.streaming = streaming
        self.config = config
        self.drop_last = drop_last
        self.load_dataset_kwargs = load_dataset_kwargs or {}

    def _open(self) -> Iterable[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "split": self.split,
            "streaming": self.streaming,
            **self.load_dataset_kwargs,
        }
        if self.config is not None:
            return load_dataset(self.path, self.config, **kwargs)
        return load_dataset(self.path, **kwargs)

    def _iter_token_docs(self) -> Iterable[list[int]]:
        worker_id, num_workers = _worker_shard()
        for i, row in enumerate(self._open()):
            if i % num_workers != worker_id:
                continue
            if self.text_field not in row:
                raise KeyError(
                    f"row missing field {self.text_field!r}; got keys {sorted(row.keys())}",
                )
            text = row[self.text_field]
            if not isinstance(text, str):
                raise TypeError(
                    f"field {self.text_field!r} is not a string in row; got {type(text).__name__}",
                )
            if not text:
                continue
            yield self.tokenizer.encode(text)

    def __iter__(self) -> Iterator[PackedBatch]:
        return pack_into_batch(
            self._iter_token_docs(),
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            drop_last=self.drop_last,
        )
