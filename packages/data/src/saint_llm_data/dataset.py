"""File-backed iterable datasets that produce packed training batches.

``TextFileDataset`` is the v0.1 workhorse: streams lines (plain text or JSONL
with a ``text`` field) through a ``Tokenizer`` and yields fixed-shape
``PackedBatch`` windows ready to feed a training step.

Subclasses ``torch.utils.data.IterableDataset`` with document-level
sharding: when wrapped in a DataLoader with ``num_workers > 0``, each
worker handles 1/N of the source documents (round-robin by line index).
Tokenization happens inside the worker so CPU work parallelizes; only the
final batched tensors cross the worker boundary.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from torch.utils.data import IterableDataset, get_worker_info

from saint_llm_data.packing import PackedBatch, pack_into_batch
from saint_llm_data.tokenizer import Tokenizer


def _iter_text_lines(path: Path, *, jsonl: bool, json_field: str) -> Iterator[str]:
    """Yield one document string per file line.

    Plain mode: each line is a document (empty lines skipped).
    JSONL mode: each line is a JSON object; ``json_field`` extracts the text.
    """
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if jsonl:
                obj = json.loads(line)
                if json_field not in obj:
                    raise KeyError(
                        f"JSONL line missing field {json_field!r} in {path.name}; got keys {list(obj.keys())}",
                    )
                yield str(obj[json_field])
            else:
                yield line


def _worker_shard() -> tuple[int, int]:
    """Return ``(worker_id, num_workers)`` from the current torch DataLoader worker.

    Returns ``(0, 1)`` outside of a DataLoader worker (single-process iteration).
    """
    info = get_worker_info()
    if info is None:
        return 0, 1
    return int(info.id), int(info.num_workers)


class TextFileDataset(IterableDataset):
    """Iterable dataset that streams text → tokens → packed training batches.

    Args:
        path: file to read.
        tokenizer: anything implementing the ``Tokenizer`` Protocol.
        seq_len: window length per row.
        batch_size: number of rows per emitted batch.
        jsonl: when True, treat each line as a JSON object and pull
            ``json_field`` from it.
        json_field: field name in JSONL mode (default ``"text"``).
        drop_last: forward-compatible with ``pack_into_batch``.

    Iteration is one-shot — re-iterate by constructing a new ``TextFileDataset``
    or wrapping in a loop (no internal cursor caching, no shuffling).
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        *,
        seq_len: int,
        batch_size: int = 1,
        jsonl: bool = False,
        json_field: str = "text",
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.jsonl = jsonl
        self.json_field = json_field
        self.drop_last = drop_last

    def _iter_token_docs(self) -> Iterable[list[int]]:
        worker_id, num_workers = _worker_shard()
        lines = _iter_text_lines(self.path, jsonl=self.jsonl, json_field=self.json_field)
        for i, text in enumerate(lines):
            if i % num_workers != worker_id:
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
