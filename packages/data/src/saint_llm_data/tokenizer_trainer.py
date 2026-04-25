"""ByteLevel BBPE tokenizer training on top of the ``tokenizers`` library.

Sticks to the GPT-2 / Llama recipe (ByteLevel pre-tokenizer + decoder,
ByteLevel.alphabet() seed) so the tokenizer handles arbitrary unicode without
unicode normalization. ``train_bbpe`` returns a loaded ``HFTokenizer`` ready
to plug into ``TextFileDataset`` / ``HuggingFaceTextDataset``.

This is the v0.1 BBPE — TOK-01..06 polish (per-script vocab allocation,
forced UK char inclusion, dropout) layers on top via ``BpeTrainer`` kwargs
the caller passes through.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer as _BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers, processors, trainers

from saint_llm_data.tokenizer import HFTokenizer

# AUGMENTATIONS TOK-04 — reserved control slots in tokenizer.TokenizerSlots.
# Defaults match cfg.tiny / cfg.v4_flash boilerplate. Override via special_tokens.
DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = (
    "<pad>",
    "<|endoftext|>",
    "<|bos|>",
    "<unk>",
    "<|image_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|video_start|>",
    "<|video_end|>",
    "<|think_start|>",
    "<|think_end|>",
)


def train_bbpe(
    corpus: Iterable[str],
    output_path: str | Path,
    *,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    dropout: float | None = 0.1,
    special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    show_progress: bool = False,
    eos_token: str = "<|endoftext|>",
    pad_token: str = "<pad>",
) -> HFTokenizer:
    """Train a ByteLevel BBPE on ``corpus`` and save to ``output_path``.

    Args:
        corpus: iterable of raw strings (one document per element). The
            tokenizers library buffers internally; large corpora can stream
            without preloading.
        output_path: where to write ``tokenizer.json``. Parent dirs created.
        vocab_size: target vocabulary size. Per AUGMENTATIONS TOK-01,
            the saint-llm production target is 131072.
        min_frequency: minimum token frequency to enter the vocab.
        dropout: BPE-dropout (TOK-06). ``None`` disables; default 0.1.
        special_tokens: tokens reserved upfront — guaranteed to land at the
            lowest IDs in encoding order. Caller should include pad/eos/bos
            and any control tokens reserved for the model.
        show_progress: forward to BpeTrainer (set True for CLI runs).
        eos_token / pad_token: which special token names the returned
            ``HFTokenizer`` should report as ``eos_token_id`` / ``pad_token_id``.

    Returns:
        loaded ``HFTokenizer`` pointing at ``output_path``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = _BackendTokenizer(
        models.BPE(unk_token="<unk>", dropout=dropout),
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )

    backend.train_from_iterator(corpus, trainer=trainer)

    # Drop dropout for inference — keeps decoding deterministic. (Training
    # tokenizer instances should re-enable it if used during model training.)
    backend.model = models.BPE(
        vocab=backend.get_vocab(),
        merges=_extract_merges(backend),
        unk_token="<unk>",
        dropout=None,
    )
    backend.save(str(output_path))
    return HFTokenizer.from_file(output_path, eos_token=eos_token, pad_token=pad_token)


def _extract_merges(backend: _BackendTokenizer) -> list[tuple[str, str]]:
    """Pull the BPE merges list out of a trained tokenizer's serialized form."""
    blob = json.loads(backend.to_str())
    merges_raw: list[str | list[str]] = blob.get("model", {}).get("merges", [])
    merges: list[tuple[str, str]] = []
    for m in merges_raw:
        if isinstance(m, list) and len(m) == 2:
            merges.append((m[0], m[1]))
        elif isinstance(m, str):
            a, _, b = m.partition(" ")
            merges.append((a, b))
    return merges
