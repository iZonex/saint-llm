"""Multimodal data plumbing — pack text + vision + audio features.

Native multimodal pretraining (per ADR-0007) requires a data
representation where each training window carries both:

* a token sequence with placeholder slots (``<|image_pad|>``,
  ``<|audio_start|>..<|audio_end|>``) at positions where modality
  features should fill in
* aligned feature tensors that the model's ``ModalityProjector``
  splices into the embedding stream at those placeholder positions

This module supplies the data shape:

* :class:`MultimodalExample` — one (prompt, response, image_features,
  audio_features) tuple
* :class:`MultimodalPackedBatch` — packed window with token IDs,
  loss mask, vision_features, audio_features, segment_ids
* :func:`encode_multimodal` — encode one example into token IDs +
  parallel feature tensors
* :func:`pack_multimodal_examples` — stream packed windows

Encoders themselves (SigLIP-2 / Whisper / etc.) live in
``saint_llm_core.multimodal``; this module assumes feature tensors
are already computed by the caller.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import torch
from torch import Tensor

from saint_llm_data.tokenizer import Tokenizer


@dataclass(frozen=True)
class MultimodalExample:
    """One multimodal training example.

    Attributes:
        prompt:           prompt text (loss-masked).
        response:         response text (loss-active).
        image_features:   tuple of ``(n_patches_i, vision_dim)`` tensors,
            one per image. The number of placeholder ``<|image_pad|>``
            tokens emitted equals ``sum(t.shape[0] for t in image_features)``.
        audio_features:   tuple of ``(n_audio_tokens_j, audio_dim)`` tensors,
            one per audio clip. Audio tokens are bracketed by
            ``<|audio_start|>...<|audio_end|>``.
    """

    prompt: str
    response: str
    image_features: tuple[Tensor, ...] = ()
    audio_features: tuple[Tensor, ...] = ()


@dataclass(frozen=True)
class MultimodalPackedBatch:
    """Packed multimodal training window.

    Attributes:
        tokens:          ``(B, T)`` long — IDs including placeholders.
        loss_mask:       ``(B, T)`` long — 1 on response tokens.
        segment_ids:     ``(B, T)`` long — example index within row.
        vision_features: ``(N_total_image_tokens, vision_dim)`` float
            — flat concat across the batch's images. The model splices
            them in at ``<|image_pad|>`` positions row-by-row.
        audio_features:  ``(N_total_audio_tokens, audio_dim)`` float —
            same idea for audio.
    """

    tokens: Tensor
    loss_mask: Tensor
    segment_ids: Tensor
    vision_features: Tensor | None = None
    audio_features: Tensor | None = None


@dataclass(frozen=True)
class TokenSlots:
    """Special token IDs the multimodal packer needs.

    Resolve from the tokenizer once, pass to ``encode_multimodal`` /
    ``pack_multimodal_examples`` so the packer doesn't have to re-look
    them up per call.
    """

    image_pad: int
    audio_start: int
    audio_end: int

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Tokenizer,
        *,
        image_pad: str = "<|image_pad|>",
        audio_start: str = "<|audio_start|>",
        audio_end: str = "<|audio_end|>",
    ) -> TokenSlots:
        backend = getattr(tokenizer, "_backend", None)
        if backend is None:
            raise ValueError(
                "tokenizer must expose a tokenizers backend "
                "(HFTokenizer); got "
                f"{type(tokenizer).__name__}",
            )
        ids = {
            "image_pad": backend.token_to_id(image_pad),
            "audio_start": backend.token_to_id(audio_start),
            "audio_end": backend.token_to_id(audio_end),
        }
        for name, val in ids.items():
            if val is None:
                raise ValueError(
                    f"tokenizer is missing the special token {name}",
                )
        return cls(
            image_pad=int(ids["image_pad"]),
            audio_start=int(ids["audio_start"]),
            audio_end=int(ids["audio_end"]),
        )


def encode_multimodal(
    example: MultimodalExample,
    tokenizer: Tokenizer,
    slots: TokenSlots,
    *,
    append_eos: bool = True,
) -> tuple[list[int], list[int]]:
    """Encode one example into ``(token_ids, loss_mask)``.

    The returned token sequence interleaves prompt text (loss-masked),
    image placeholder slots (one ``<|image_pad|>`` per image patch,
    loss-masked), audio placeholder slots (bracketed by start/end,
    loss-masked), and response tokens (loss-active).

    Layout: ``[prompt_text] [image_0 patches] ... [image_N patches]
    [audio_0 start..end] ... [audio_M start..end] [response_text]
    [eos]``.

    The loss mask is 0 on prompt + all modality slots, 1 on response
    + EOS (so the model learns to stop after the response).
    """
    prompt_ids = tokenizer.encode(example.prompt)
    response_ids = tokenizer.encode(example.response)

    token_ids: list[int] = list(prompt_ids)
    loss_mask: list[int] = [0] * len(prompt_ids)

    # Image placeholders: one <|image_pad|> per patch, all images concat.
    for img in example.image_features:
        n_patches = int(img.shape[0])
        token_ids.extend([slots.image_pad] * n_patches)
        loss_mask.extend([0] * n_patches)

    # Audio placeholders: each clip bracketed by start/end.
    for clip in example.audio_features:
        n_audio_tokens = int(clip.shape[0])
        token_ids.append(slots.audio_start)
        loss_mask.append(0)
        token_ids.extend([slots.audio_start] * n_audio_tokens)
        loss_mask.extend([0] * n_audio_tokens)
        token_ids.append(slots.audio_end)
        loss_mask.append(0)

    # Response.
    token_ids.extend(response_ids)
    loss_mask.extend([1] * len(response_ids))

    if append_eos:
        token_ids.append(tokenizer.eos_token_id)
        loss_mask.append(1)

    return token_ids, loss_mask


@dataclass
class _PackBuffer:
    """Mutable scratch buffer used while packing examples into windows."""

    tokens: list[int] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)
    segments: list[int] = field(default_factory=list)
    image_feats: list[Tensor] = field(default_factory=list)
    audio_feats: list[Tensor] = field(default_factory=list)


def pack_multimodal_examples(
    examples: Iterable[MultimodalExample],
    tokenizer: Tokenizer,
    slots: TokenSlots,
    *,
    seq_len: int,
    pad_token_id: int = 0,
    append_eos: bool = True,
    drop_last: bool = True,
) -> Iterator[MultimodalPackedBatch]:
    """Stream packed ``(1, seq_len)`` multimodal windows.

    Concat-and-chunk just like text SFT packing: each example's
    encoded tokens + features are appended to a running buffer; when
    the token buffer reaches ``seq_len``, one window is emitted with
    the corresponding feature slice.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive; got {seq_len}")

    buf = _PackBuffer()
    next_segment = 0

    def _emit_window() -> MultimodalPackedBatch:
        # Take seq_len tokens; the corresponding feature tensors are
        # whatever has been queued so far for this window. Tail
        # features beyond the cut belong to the *next* example, but
        # for simplicity v0.0 emits all queued features per window
        # (they get spliced in at their placeholder positions; if a
        # placeholder gets cut by the window boundary, those features
        # still go through but apply at later windows). The packer
        # produces correct windows when example boundaries align with
        # seq_len; for misalignment, callers should set seq_len large
        # enough to fit one example.
        win_tokens = buf.tokens[:seq_len]
        win_mask = buf.mask[:seq_len]
        win_seg = buf.segments[:seq_len]
        # Reset feature accumulator on emit; the trainer will recompute
        # per-window splices from the placeholder positions.
        vision_feats = (
            torch.cat(buf.image_feats, dim=0) if buf.image_feats else None
        )
        audio_feats = (
            torch.cat(buf.audio_feats, dim=0) if buf.audio_feats else None
        )
        result = MultimodalPackedBatch(
            tokens=torch.tensor(win_tokens, dtype=torch.long).unsqueeze(0),
            loss_mask=torch.tensor(win_mask, dtype=torch.long).unsqueeze(0),
            segment_ids=torch.tensor(win_seg, dtype=torch.long).unsqueeze(0),
            vision_features=vision_feats,
            audio_features=audio_feats,
        )
        buf.tokens = buf.tokens[seq_len:]
        buf.mask = buf.mask[seq_len:]
        buf.segments = buf.segments[seq_len:]
        buf.image_feats = []
        buf.audio_feats = []
        return result

    for example in examples:
        ids, mask = encode_multimodal(example, tokenizer, slots, append_eos=append_eos)
        if not ids:
            continue
        buf.tokens.extend(ids)
        buf.mask.extend(mask)
        buf.segments.extend([next_segment] * len(ids))
        buf.image_feats.extend(example.image_features)
        buf.audio_feats.extend(example.audio_features)
        next_segment += 1
        while len(buf.tokens) >= seq_len:
            yield _emit_window()

    if buf.tokens and not drop_last:
        pad_n = seq_len - len(buf.tokens)
        last_seg = buf.segments[-1] if buf.segments else 0
        buf.tokens.extend([pad_token_id] * pad_n)
        buf.mask.extend([0] * pad_n)
        buf.segments.extend([last_seg] * pad_n)
        yield _emit_window()
