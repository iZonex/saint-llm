"""ByteLevel BBPE tokenizer training on top of the ``tokenizers`` library.

Sticks to the GPT-2 / Llama recipe (ByteLevel pre-tokenizer + decoder,
ByteLevel.alphabet() seed) so the tokenizer handles arbitrary unicode without
unicode normalization. ``train_bbpe`` returns a loaded ``HFTokenizer`` ready
to plug into ``TextFileDataset`` / ``HuggingFaceTextDataset``.

This is the v0.0 / v0.1 BBPE per AUGMENTATIONS TOK-01..06 + ADR-0004
(own BBPE 131K). The expanded ``SAINT_V0_0_SPECIAL_TOKENS`` reserves
control slots for modality markers, reasoning markers, memory verbs,
effort tiers (RL-07 / ADR-0017), and Quick Instruction tokens. The
``force_include_chars`` argument satisfies TOK-03 (UK chars and
Cyrillic four-case forms).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer as _BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers, processors, trainers

from saint_llm_data.tokenizer import HFTokenizer

# Backwards-compatible minimal set used by older tests / CLI examples.
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

# v0.0 production set per ADR-0004 (own BBPE 131K) + ADR-0007 (multimodal)
# + ADR-0008 (memory tools) + ADR-0017 (adaptive thinking effort head).
# Token slots reserved here, lazy embedding-row allocation per TOK-04 +
# TOK-05 tied embeddings.
SAINT_V0_0_SPECIAL_TOKENS: tuple[str, ...] = (
    # Core control
    "<pad>",
    "<|endoftext|>",
    "<|bos|>",
    "<unk>",
    # Multimodal (ADR-0007)
    "<|image_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|video_start|>",
    "<|video_end|>",
    # Reasoning markers (REA-01 V4 think modes baseline + REA-08 reflection)
    "<|think_start|>",
    "<|think_end|>",
    "<|reflect|>",
    "</reflect|>",
    # Adaptive thinking effort tiers (RL-07 / ADR-0017, replaces REA-05)
    "<|effort:0|>",  # low
    "<|effort:1|>",  # medium
    "<|effort:2|>",  # high
    "<|effort:3|>",  # xhigh
    "<|effort:4|>",  # max
    # Memory verbs (MEM-01 / ADR-0008, Letta-style verb-oriented memory)
    "<|memory_recall|>",
    "<|memory_save|>",
    "<|memory_result|>",
    # Quick Instruction reserved (post-training feature, slots only at v0.0)
    "<|action|>",
    "<|title|>",
    "<|query|>",
)


# Per AUGMENTATIONS TOK-03 (force-include UK chars + four-case Cyrillic).
# Caller can extend / override; these are the canonical UK additions on top
# of ByteLevel.alphabet() (which covers Latin extended, basic Cyrillic).
# The lines below contain genuine Cyrillic codepoints that visually look
# like Latin letters; that visual ambiguity is exactly the point —
# these are the codepoints we need the tokenizer to seed merges from.
SAINT_V0_0_FORCE_INCLUDE_CHARS: tuple[str, ...] = (
    # Ukrainian-specific letters
    "ї", "Ї", "є", "Є", "ґ", "Ґ", "і", "І",  # noqa: RUF001
    # Apostrophe variants critical for UK
    "'", "ʼ", "’",  # straight, modifier letter, right single quote  # noqa: RUF001
    # Cyrillic four-case markers used in UK / RU / belarussian morphology
    # (caller can extend; these are the basic set)
)


def train_bbpe(
    corpus: Iterable[str],
    output_path: str | Path,
    *,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    dropout: float | None = 0.1,
    special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    force_include_chars: Iterable[str] = (),
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
        force_include_chars: characters or short strings forcibly added to
            the initial alphabet so they're guaranteed atomic vocab
            entries. Per AUGMENTATIONS TOK-03 the saint-llm production
            recipe forces Ukrainian Cyrillic letters and apostrophe
            variants which the ByteLevel default alphabet covers as
            bytes but may not reach as merge-result tokens at smaller
            vocab budgets. See ``SAINT_V0_0_FORCE_INCLUDE_CHARS`` for the
            canonical UK-targeted set.
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

    # ByteLevel alphabet = 256 base bytes covering all of UTF-8. force_include_chars
    # extends this with extra atomic seeds. Multi-byte characters (e.g. Cyrillic)
    # are decomposed by ByteLevel pre-tokenizer to their UTF-8 byte sequence;
    # adding them here makes their multi-byte form available as a single
    # initial-alphabet entry for the BPE trainer to use as merge seed.
    initial_alphabet = list(pre_tokenizers.ByteLevel.alphabet())
    for ch in force_include_chars:
        if ch not in initial_alphabet:
            initial_alphabet.append(ch)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens),
        initial_alphabet=initial_alphabet,
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
