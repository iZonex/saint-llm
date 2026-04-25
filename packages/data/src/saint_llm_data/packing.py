"""Token-sequence packing for pretraining.

Concatenates variable-length tokenized documents into fixed-length training
windows of shape ``(B, T)`` with EOS separators between docs. Also returns
``segment_ids`` so downstream attention can mask cross-document attention if
desired (cheap to compute, cheap to ignore).

The simple "concat + chunk" recipe used by GPT-NeoX, FineWeb pretraining, and
DeepSeek's own tokenizer pipeline. Sample-level masking and FIM packing live
on top — separate modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PackedBatch:
    """One ``(B, T)`` window of packed tokens.

    Attributes:
        tokens:      ``(B, T)`` long — token IDs.
        segment_ids: ``(B, T)`` long — ascending integer per document within
            each row. Useful as a per-token attention-mask key.
    """

    tokens: Tensor
    segment_ids: Tensor

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.tokens.shape)


def pack_sequences(
    docs: Iterable[list[int]],
    *,
    seq_len: int,
    eos_token_id: int,
    pad_token_id: int = 0,
    drop_last: bool = True,
) -> Iterator[PackedBatch]:
    """Stream packed ``(1, seq_len)`` batches from a stream of tokenized docs.

    Args:
        docs: iterable of token-ID lists. Each item is one document.
        seq_len: target window length.
        eos_token_id: appended after each document so the model sees a hard
            boundary signal during training.
        pad_token_id: used to pad the final residual window when
            ``drop_last=False`` and the buffer holds < ``seq_len`` tokens.
        drop_last: if True, the trailing residual is discarded; if False, the
            last window is right-padded with ``pad_token_id``.

    Yields:
        ``PackedBatch`` of shape ``(1, seq_len)`` per emitted window. Caller
        can stack multiple yields into ``(B, seq_len)`` if desired.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive; got {seq_len}")

    buffer_tokens: list[int] = []
    buffer_segments: list[int] = []
    next_segment = 0

    for doc in docs:
        if not doc:
            continue
        buffer_tokens.extend(doc)
        buffer_segments.extend([next_segment] * len(doc))
        buffer_tokens.append(eos_token_id)
        buffer_segments.append(next_segment)
        next_segment += 1

        while len(buffer_tokens) >= seq_len:
            window = buffer_tokens[:seq_len]
            seg = buffer_segments[:seq_len]
            buffer_tokens = buffer_tokens[seq_len:]
            buffer_segments = buffer_segments[seq_len:]
            yield PackedBatch(
                tokens=torch.tensor(window, dtype=torch.long).unsqueeze(0),
                segment_ids=torch.tensor(seg, dtype=torch.long).unsqueeze(0),
            )

    if buffer_tokens and not drop_last:
        pad_count = seq_len - len(buffer_tokens)
        last_seg = buffer_segments[-1] if buffer_segments else 0
        window = buffer_tokens + [pad_token_id] * pad_count
        seg = buffer_segments + [last_seg] * pad_count
        yield PackedBatch(
            tokens=torch.tensor(window, dtype=torch.long).unsqueeze(0),
            segment_ids=torch.tensor(seg, dtype=torch.long).unsqueeze(0),
        )


def pack_into_batch(
    docs: Iterable[list[int]],
    *,
    batch_size: int,
    seq_len: int,
    eos_token_id: int,
    pad_token_id: int = 0,
    drop_last: bool = True,
) -> Iterator[PackedBatch]:
    """Group ``pack_sequences`` output into ``(batch_size, seq_len)`` batches."""
    rows: list[PackedBatch] = []
    for window in pack_sequences(
        docs,
        seq_len=seq_len,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        drop_last=drop_last,
    ):
        rows.append(window)
        if len(rows) == batch_size:
            yield PackedBatch(
                tokens=torch.cat([r.tokens for r in rows], dim=0),
                segment_ids=torch.cat([r.segment_ids for r in rows], dim=0),
            )
            rows = []
    if rows and not drop_last:
        # Pad the final partial batch with zero rows.
        empty_pad = batch_size - len(rows)
        if empty_pad > 0:
            zero_tokens = torch.full((empty_pad, seq_len), pad_token_id, dtype=torch.long)
            zero_segs = torch.full((empty_pad, seq_len), -1, dtype=torch.long)
            tokens = torch.cat([*[r.tokens for r in rows], zero_tokens], dim=0)
            segs = torch.cat([*[r.segment_ids for r in rows], zero_segs], dim=0)
        else:
            tokens = torch.cat([r.tokens for r in rows], dim=0)
            segs = torch.cat([r.segment_ids for r in rows], dim=0)
        yield PackedBatch(tokens=tokens, segment_ids=segs)
