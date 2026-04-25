"""CLI: train a ByteLevel BBPE tokenizer on a text corpus.

Sources:
* ``--corpus path`` — plain-text file (one document per line).
* ``--hf-dataset path`` — stream from HF Hub (text-field configurable).
* ``--corpus`` and ``--hf-dataset`` are mutually exclusive.

Examples:
    uv run python experiments/train_tokenizer.py \\
        --corpus data/corpus.txt --vocab-size 32000 --output tokenizer.json

    uv run python experiments/train_tokenizer.py \\
        --hf-dataset roneneldan/TinyStories --hf-split "train[:5%]" \\
        --vocab-size 32000 --output tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from datasets import load_dataset
from saint_llm_data import DEFAULT_SPECIAL_TOKENS, train_bbpe


def _iter_text_file(path: Path, *, jsonl: bool, json_field: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if jsonl:
                obj = json.loads(line)
                if json_field in obj and isinstance(obj[json_field], str):
                    yield obj[json_field]
            else:
                yield line


def _iter_hf_dataset(
    name: str,
    *,
    split: str,
    text_field: str,
    streaming: bool,
    max_rows: int | None,
) -> Iterable[str]:
    ds = load_dataset(name, split=split, streaming=streaming)
    for i, row in enumerate(ds):
        if max_rows is not None and i >= max_rows:
            break
        text = row.get(text_field)
        if isinstance(text, str) and text:
            yield text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="path to write tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus", type=Path,
                     help="plain-text or JSONL file (one document per line).")
    src.add_argument("--hf-dataset", type=str,
                     help="HF Hub dataset path, e.g. roneneldan/TinyStories.")

    parser.add_argument("--jsonl", action="store_true",
                        help="treat --corpus as JSONL (uses --json-field).")
    parser.add_argument("--json-field", default="text")

    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-field", default="text")
    parser.add_argument("--hf-streaming", action="store_true", default=True)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="cap rows pulled from --hf-dataset; useful for "
                             "tokenizer iteration speed.")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress BpeTrainer progress bar.")
    args = parser.parse_args()

    if args.corpus is not None:
        corpus = _iter_text_file(args.corpus, jsonl=args.jsonl, json_field=args.json_field)
        source = str(args.corpus)
    else:
        corpus = _iter_hf_dataset(
            args.hf_dataset,
            split=args.hf_split,
            text_field=args.hf_text_field,
            streaming=args.hf_streaming,
            max_rows=args.max_rows,
        )
        source = f"{args.hf_dataset}::{args.hf_split}"

    print(f"Training BBPE | source={source} | vocab_size={args.vocab_size} | "
          f"min_freq={args.min_frequency} | dropout={args.dropout}")
    print(f"Output → {args.output}")
    print()

    tok = train_bbpe(
        corpus,
        args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        dropout=args.dropout,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        show_progress=not args.quiet,
    )

    print()
    print(f"Done. Final vocab size: {tok.vocab_size}")
    print(f"  eos_token_id = {tok.eos_token_id}")
    print(f"  pad_token_id = {tok.pad_token_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
