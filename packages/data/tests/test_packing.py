"""Sequence packing: concat semantics, EOS placement, segment IDs, edge cases."""

from __future__ import annotations

import pytest
import torch
from saint_llm_data import PackedBatch, pack_into_batch, pack_sequences


def _docs(*lengths: int) -> list[list[int]]:
    """Helper: make docs filled with sequential ints so ordering is debuggable."""
    out = []
    next_val = 1
    for n in lengths:
        out.append(list(range(next_val, next_val + n)))
        next_val += 1000
    return out


def test_pack_sequences_yields_full_windows() -> None:
    docs = _docs(3, 4)  # 3 + EOS + 4 + EOS = 9 tokens; seq_len=4 → 2 windows + leftover
    eos = 99
    windows = list(pack_sequences(docs, seq_len=4, eos_token_id=eos, drop_last=True))
    assert len(windows) == 2
    for w in windows:
        assert isinstance(w, PackedBatch)
        assert w.tokens.shape == (1, 4)
        assert w.segment_ids.shape == (1, 4)


def test_pack_sequences_appends_eos_after_each_doc() -> None:
    docs = [[1, 2], [3]]
    eos = 99
    windows = list(pack_sequences(docs, seq_len=3, eos_token_id=eos, drop_last=False))
    flat: list[int] = []
    for w in windows:
        flat.extend(w.tokens[0].tolist())
    # First doc: 1 2 EOS, second doc: 3 EOS
    assert flat[:5] == [1, 2, eos, 3, eos]


def test_pack_sequences_segment_ids_increment_per_doc() -> None:
    docs = [[10, 11], [20]]
    eos = 99
    windows = list(pack_sequences(docs, seq_len=2, eos_token_id=eos, drop_last=False))
    seg_flat: list[int] = []
    for w in windows:
        seg_flat.extend(w.segment_ids[0].tolist())
    # Doc 0 spans tokens [10, 11, EOS] → seg 0; Doc 1 spans [20, EOS] → seg 1.
    assert seg_flat[:3] == [0, 0, 0]
    assert seg_flat[3:5] == [1, 1]


def test_pack_sequences_drops_residual_when_drop_last() -> None:
    """With drop_last=True, no padded final window is emitted."""
    docs = [[1, 2, 3]]
    windows = list(pack_sequences(docs, seq_len=10, eos_token_id=99, drop_last=True))
    assert windows == []


def test_pack_sequences_pads_residual_when_drop_last_false() -> None:
    docs = [[1, 2, 3]]
    eos = 99
    pad = 0
    windows = list(pack_sequences(
        docs, seq_len=10, eos_token_id=eos, pad_token_id=pad, drop_last=False,
    ))
    assert len(windows) == 1
    tokens = windows[0].tokens[0].tolist()
    assert tokens[:4] == [1, 2, 3, eos]
    assert tokens[4:] == [pad] * 6


def test_pack_sequences_skips_empty_docs() -> None:
    docs = [[], [1, 2], [], [3]]
    eos = 99
    windows = list(pack_sequences(docs, seq_len=2, eos_token_id=eos, drop_last=False))
    flat = []
    for w in windows:
        flat.extend(w.tokens[0].tolist())
    # Empty docs contribute nothing — not even an EOS.
    assert flat[:5] == [1, 2, eos, 3, eos]


def test_pack_sequences_rejects_non_positive_seq_len() -> None:
    with pytest.raises(ValueError, match="seq_len must be positive"):
        list(pack_sequences([[1, 2]], seq_len=0, eos_token_id=99))


def test_pack_into_batch_groups_rows() -> None:
    docs = _docs(8, 8, 8, 8)  # plenty of windows
    eos = 99
    batches = list(pack_into_batch(
        docs, batch_size=4, seq_len=4, eos_token_id=eos, drop_last=True,
    ))
    assert all(b.tokens.shape == (4, 4) for b in batches)
    assert all(b.segment_ids.shape == (4, 4) for b in batches)


def test_pack_into_batch_pads_partial_final_batch() -> None:
    """Few windows with batch_size > windows + drop_last=False → pad with -1 segments."""
    eos = 99
    # Two short docs → one full window + one short residual = 2 windows total.
    docs = [[1, 2, 3], [4, 5, 6]]  # 3 + EOS + 3 + EOS = 8 tokens; seq_len=4 → 2 windows.
    batches = list(pack_into_batch(
        docs, batch_size=4, seq_len=4, eos_token_id=eos, drop_last=False,
    ))
    assert len(batches) == 1
    final = batches[0]
    assert final.tokens.shape == (4, 4)
    # The two trailing padded rows should have segment_id == -1.
    assert (final.segment_ids[2:] == -1).all()
    # And the two real rows should not.
    assert (final.segment_ids[:2] != -1).any()


def test_packed_batch_is_tensor_pair() -> None:
    docs = [[1, 2, 3, 4, 5]]
    batches = list(pack_sequences(docs, seq_len=4, eos_token_id=99, drop_last=True))
    assert len(batches) == 1
    assert isinstance(batches[0].tokens, torch.Tensor)
    assert isinstance(batches[0].segment_ids, torch.Tensor)
    assert batches[0].shape == (1, 4)
