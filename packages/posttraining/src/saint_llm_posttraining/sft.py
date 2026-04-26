"""SFT data plumbing — encode (prompt, response) pairs into packed batches.

The SFT loss is plain next-token cross-entropy *masked* so only response
tokens contribute. Prompt tokens are still in the input stream (the model
must condition on them) but are not optimized against. This is the
"keep-response, mask-prompt" recipe used by every modern instruction-tune
pipeline (Llama, DeepSeek, Mistral all do this in some form).

Pipeline:

    SFTExample(prompt, response[, system])
        ↓ encode_sft
    (token_ids, loss_mask)        # 0 on prompt/system, 1 on response (+EOS)
        ↓ pack_sft_examples
    SFTPackedBatch(tokens, loss_mask, segment_ids)
        ↓ sft_cross_entropy(out, batch)
    scalar loss

Packing concatenates examples back-to-back (just like pretraining) so the
window stays full. ``segment_ids`` rises monotonically within each row so
downstream attention can mask cross-example attention if it wants to.

Shape: ``pack_sft_examples`` yields ``(1, seq_len)`` rows;
``pack_sft_into_batch`` groups them into ``(batch_size, seq_len)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from saint_llm_data.tokenizer import Tokenizer
from torch import Tensor


@dataclass(frozen=True)
class SFTExample:
    """A single (prompt, response) supervision pair.

    Attributes:
        prompt:   the user-visible question / instruction. Loss-masked out.
        response: the target completion. The only thing the model is trained
            to produce.
        system:   optional system preface joined before the prompt with a
            newline. Loss-masked out alongside the prompt.
    """

    prompt: str
    response: str
    system: str | None = None


@dataclass(frozen=True)
class SFTPackedBatch:
    """One window of packed SFT tokens.

    Attributes:
        tokens:      ``(B, T)`` long — token IDs.
        loss_mask:   ``(B, T)`` long — 1 on response tokens (incl. EOS when
            appended), 0 on prompt/system tokens and right-pad. Used by
            ``sft_cross_entropy`` to gate which positions contribute.
        segment_ids: ``(B, T)`` long — ascending integer per example within
            each row. Mirrors ``PackedBatch.segment_ids`` from pretraining.
    """

    tokens: Tensor
    loss_mask: Tensor
    segment_ids: Tensor

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.tokens.shape)


def encode_sft(
    example: SFTExample,
    tokenizer: Tokenizer,
    *,
    append_eos: bool = True,
) -> tuple[list[int], list[int]]:
    """Tokenize one example into ``(token_ids, loss_mask)`` of equal length.

    Loss-mask is 0 on prompt/system tokens, 1 on response tokens. When
    ``append_eos`` is true the tokenizer's EOS id is appended with mask=1
    so the model learns to stop.
    """
    prefix_text = (example.system + "\n" if example.system else "") + example.prompt
    prefix_ids = tokenizer.encode(prefix_text)
    response_ids = tokenizer.encode(example.response)

    token_ids = list(prefix_ids) + list(response_ids)
    loss_mask = [0] * len(prefix_ids) + [1] * len(response_ids)

    if append_eos:
        token_ids.append(tokenizer.eos_token_id)
        loss_mask.append(1)

    return token_ids, loss_mask


def pack_sft_examples(
    examples: Iterable[SFTExample],
    tokenizer: Tokenizer,
    *,
    seq_len: int,
    pad_token_id: int = 0,
    append_eos: bool = True,
    drop_last: bool = True,
) -> Iterator[SFTPackedBatch]:
    """Stream packed ``(1, seq_len)`` SFT batches.

    Concat-and-chunk just like pretraining packing: each example appends
    its (token_ids, loss_mask) to a running buffer and ``seq_len``-sized
    windows are emitted as soon as available. The trailing residual is
    dropped unless ``drop_last=False``, in which case it is right-padded
    with ``pad_token_id`` and mask=0 (pad does not contribute to loss).
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive; got {seq_len}")

    buf_tokens: list[int] = []
    buf_mask: list[int] = []
    buf_seg: list[int] = []
    next_segment = 0

    for example in examples:
        ids, mask = encode_sft(example, tokenizer, append_eos=append_eos)
        if not ids:
            continue
        buf_tokens.extend(ids)
        buf_mask.extend(mask)
        buf_seg.extend([next_segment] * len(ids))
        next_segment += 1

        while len(buf_tokens) >= seq_len:
            yield SFTPackedBatch(
                tokens=torch.tensor(buf_tokens[:seq_len], dtype=torch.long).unsqueeze(0),
                loss_mask=torch.tensor(buf_mask[:seq_len], dtype=torch.long).unsqueeze(0),
                segment_ids=torch.tensor(buf_seg[:seq_len], dtype=torch.long).unsqueeze(0),
            )
            buf_tokens = buf_tokens[seq_len:]
            buf_mask = buf_mask[seq_len:]
            buf_seg = buf_seg[seq_len:]

    if buf_tokens and not drop_last:
        pad_n = seq_len - len(buf_tokens)
        last_seg = buf_seg[-1] if buf_seg else 0
        yield SFTPackedBatch(
            tokens=torch.tensor(buf_tokens + [pad_token_id] * pad_n, dtype=torch.long).unsqueeze(0),
            loss_mask=torch.tensor(buf_mask + [0] * pad_n, dtype=torch.long).unsqueeze(0),
            segment_ids=torch.tensor(buf_seg + [last_seg] * pad_n, dtype=torch.long).unsqueeze(0),
        )


def pack_sft_into_batch(
    examples: Iterable[SFTExample],
    tokenizer: Tokenizer,
    *,
    batch_size: int,
    seq_len: int,
    pad_token_id: int = 0,
    append_eos: bool = True,
    drop_last: bool = True,
) -> Iterator[SFTPackedBatch]:
    """Group ``pack_sft_examples`` output into ``(batch_size, seq_len)`` batches."""
    rows: list[SFTPackedBatch] = []
    for window in pack_sft_examples(
        examples,
        tokenizer,
        seq_len=seq_len,
        pad_token_id=pad_token_id,
        append_eos=append_eos,
        drop_last=drop_last,
    ):
        rows.append(window)
        if len(rows) == batch_size:
            yield SFTPackedBatch(
                tokens=torch.cat([r.tokens for r in rows], dim=0),
                loss_mask=torch.cat([r.loss_mask for r in rows], dim=0),
                segment_ids=torch.cat([r.segment_ids for r in rows], dim=0),
            )
            rows = []

    if rows and not drop_last:
        empty_pad = batch_size - len(rows)
        if empty_pad > 0:
            zero_tokens = torch.full((empty_pad, seq_len), pad_token_id, dtype=torch.long)
            zero_mask = torch.zeros((empty_pad, seq_len), dtype=torch.long)
            zero_seg = torch.full((empty_pad, seq_len), -1, dtype=torch.long)
            yield SFTPackedBatch(
                tokens=torch.cat([*[r.tokens for r in rows], zero_tokens], dim=0),
                loss_mask=torch.cat([*[r.loss_mask for r in rows], zero_mask], dim=0),
                segment_ids=torch.cat([*[r.segment_ids for r in rows], zero_seg], dim=0),
            )
        else:
            yield SFTPackedBatch(
                tokens=torch.cat([r.tokens for r in rows], dim=0),
                loss_mask=torch.cat([r.loss_mask for r in rows], dim=0),
                segment_ids=torch.cat([r.segment_ids for r in rows], dim=0),
            )


def sft_cross_entropy(
    out: dict[str, Tensor | list[Tensor]],
    batch: SFTPackedBatch,
) -> Tensor:
    """Masked next-token cross-entropy on response positions only.

    Position ``t`` predicts token ``t+1``. The CE term at position ``t`` is
    kept iff ``batch.loss_mask[:, t+1] == 1`` — i.e. the *target* is a
    response token. The loss is the mean CE over kept positions; if no
    positions are kept (degenerate batch with all-prompt rows), the
    denominator is clamped to 1 so the result is a clean zero rather than
    NaN.
    """
    logits = out["logits"]
    if not isinstance(logits, Tensor):
        raise TypeError("out['logits'] must be a Tensor")

    pred_logits = logits[:, :-1]
    targets = batch.tokens[:, 1:].to(device=pred_logits.device)
    mask = batch.loss_mask[:, 1:].to(device=pred_logits.device, dtype=pred_logits.dtype)

    ce = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(mask)
    masked = ce * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom
