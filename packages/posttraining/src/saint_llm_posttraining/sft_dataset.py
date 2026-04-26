"""SFT dataset adapters — stream JSONL ``{prompt, response}`` rows.

Composes with the existing SFT math layer (:mod:`sft`) so the trainer
driver doesn't need to know about JSONL format details.

Two readers ship:

* :class:`JsonlSFTDataset` — reads a local JSONL file row-by-row.
* :class:`HFSFTDataset` — streams from an HF Hub dataset; each row
  must have ``prompt_field`` and ``response_field`` columns.

Both yield :class:`SFTPackedBatch` packed windows ready for
:func:`sft_cross_entropy`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from saint_llm_data.tokenizer import Tokenizer
from torch.utils.data import IterableDataset, get_worker_info

from saint_llm_posttraining.sft import (
    SFTExample,
    SFTPackedBatch,
    pack_sft_into_batch,
)


def _iter_jsonl_examples(
    path: Path,
    *,
    prompt_field: str,
    response_field: str,
    system_field: str | None,
) -> Iterator[SFTExample]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get(prompt_field)
            response = obj.get(response_field)
            if not isinstance(prompt, str) or not isinstance(response, str):
                continue
            system = (
                obj.get(system_field)
                if system_field and isinstance(obj.get(system_field), str)
                else None
            )
            yield SFTExample(prompt=prompt, response=response, system=system)


class JsonlSFTDataset(IterableDataset):
    """Stream packed SFT batches from a JSONL file.

    Each line must JSON-decode to an object with a ``prompt`` and
    ``response`` field (names configurable). Optional ``system`` field
    provides a system prompt that is loss-masked alongside the prompt.

    Multi-worker friendly: each worker reads disjoint lines via
    line-index sharding.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        tokenizer: Tokenizer,
        seq_len: int,
        batch_size: int = 1,
        prompt_field: str = "prompt",
        response_field: str = "response",
        system_field: str | None = "system",
        append_eos: bool = True,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.system_field = system_field
        self.append_eos = append_eos
        self.drop_last = drop_last
        if not self.path.exists():
            raise FileNotFoundError(f"SFT JSONL file not found: {self.path}")

    def _examples_for_worker(self) -> Iterable[SFTExample]:
        info = get_worker_info()
        all_examples = _iter_jsonl_examples(
            self.path,
            prompt_field=self.prompt_field,
            response_field=self.response_field,
            system_field=self.system_field,
        )
        if info is None:
            yield from all_examples
            return
        for idx, ex in enumerate(all_examples):
            if idx % info.num_workers == info.id:
                yield ex

    def __iter__(self) -> Iterator[SFTPackedBatch]:
        yield from pack_sft_into_batch(
            self._examples_for_worker(),
            self.tokenizer,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            append_eos=self.append_eos,
            drop_last=self.drop_last,
        )
