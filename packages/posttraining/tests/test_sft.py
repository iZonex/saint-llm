"""Tests for SFT data plumbing.

Use ``CharTokenizer`` (deterministic codepoint-as-id) so we can easily
predict prompt/response boundaries from the input strings — saves us from
having to train or download a real BBPE for hermetic tests.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from saint_llm_data.tokenizer import CharTokenizer
from saint_llm_posttraining import (
    SFTExample,
    SFTPackedBatch,
    encode_sft,
    pack_sft_examples,
    pack_sft_into_batch,
    sft_cross_entropy,
)


def _tok() -> CharTokenizer:
    # Reasonable BMP-only fallback; ids are codepoint+base_vocab.
    return CharTokenizer(base_vocab=16, unicode_max=0x10000)


def test_encode_sft_lengths_and_mask() -> None:
    tok = _tok()
    ex = SFTExample(prompt="abc", response="DE")
    ids, mask = encode_sft(ex, tok, append_eos=True)

    # 3 prompt + 2 response + 1 eos = 6
    assert len(ids) == 6
    assert len(mask) == 6
    # First 3 are prompt, masked out.
    assert mask[:3] == [0, 0, 0]
    # Response (+EOS) is included in the loss.
    assert mask[3:] == [1, 1, 1]
    # The EOS slot is the tokenizer's eos id.
    assert ids[-1] == tok.eos_token_id


def test_encode_sft_no_eos() -> None:
    tok = _tok()
    ex = SFTExample(prompt="hi", response="ok")
    ids, mask = encode_sft(ex, tok, append_eos=False)
    assert len(ids) == len(mask) == 4
    assert mask == [0, 0, 1, 1]


def test_encode_sft_with_system_prefix_is_masked() -> None:
    tok = _tok()
    ex = SFTExample(system="SYS", prompt="P", response="R")
    ids, mask = encode_sft(ex, tok, append_eos=False)
    # system "SYS" + "\n" + prompt "P" = 5 prefix chars (mask 0),
    # response "R" = 1 token (mask 1).
    assert len(ids) == 6
    assert mask == [0, 0, 0, 0, 0, 1]


def test_encode_sft_response_decodes_back() -> None:
    """Roundtrip: ids[mask==1] should decode back to response (+EOS skipped)."""
    tok = _tok()
    ex = SFTExample(prompt="hello, ", response="world!")
    ids, mask = encode_sft(ex, tok, append_eos=False)
    response_ids = [i for i, m in zip(ids, mask, strict=True) if m == 1]
    assert tok.decode(response_ids) == "world!"


def test_pack_sft_emits_expected_window_shape_and_mask() -> None:
    tok = _tok()
    examples = [SFTExample(prompt="ab", response="cd"), SFTExample(prompt="e", response="fgh")]
    # Each ex: 2 prompt + 2 response + 1 eos = 5 / 1 prompt + 3 response + 1 eos = 5
    # Total tokens: 10. seq_len=5 → 2 windows.
    windows = list(pack_sft_examples(examples, tok, seq_len=5, drop_last=True))
    assert len(windows) == 2
    for w in windows:
        assert isinstance(w, SFTPackedBatch)
        assert w.tokens.shape == (1, 5)
        assert w.loss_mask.shape == (1, 5)
        assert w.segment_ids.shape == (1, 5)

    # Window 0: example 0's full sequence, mask = [0,0,1,1,1]
    assert windows[0].loss_mask.tolist() == [[0, 0, 1, 1, 1]]
    assert windows[0].segment_ids.tolist() == [[0, 0, 0, 0, 0]]

    # Window 1: example 1's full sequence, mask = [0,1,1,1,1]
    assert windows[1].loss_mask.tolist() == [[0, 1, 1, 1, 1]]
    assert windows[1].segment_ids.tolist() == [[1, 1, 1, 1, 1]]


def test_pack_sft_drops_residual_when_drop_last_true() -> None:
    tok = _tok()
    examples = [SFTExample(prompt="abc", response="d")]  # 3+1+1 = 5 tokens
    # seq_len=4: residual of 1 token, dropped.
    out = list(pack_sft_examples(examples, tok, seq_len=4, drop_last=True))
    assert len(out) == 1
    assert out[0].tokens.shape == (1, 4)


def test_pack_sft_pads_residual_when_drop_last_false() -> None:
    tok = _tok()
    examples = [SFTExample(prompt="ab", response="c")]  # 2+1+1 = 4 tokens
    out = list(pack_sft_examples(examples, tok, seq_len=6, pad_token_id=0, drop_last=False))
    assert len(out) == 1
    w = out[0]
    assert w.tokens.shape == (1, 6)
    # Mask: 2 prompt zeros, 1 response one, 1 eos one, 2 pad zeros.
    assert w.loss_mask.tolist() == [[0, 0, 1, 1, 0, 0]]
    # Pad positions hold pad_token_id=0.
    assert w.tokens[0, -2:].tolist() == [0, 0]


def test_pack_sft_segment_ids_increment_across_examples() -> None:
    tok = _tok()
    examples = [
        SFTExample(prompt="a", response="b"),  # 1+1+1 = 3 tokens
        SFTExample(prompt="c", response="d"),  # 1+1+1 = 3 tokens
        SFTExample(prompt="e", response="f"),  # 1+1+1 = 3 tokens
    ]
    # Total 9, seq_len=9 → one window with segments 0,0,0,1,1,1,2,2,2.
    out = list(pack_sft_examples(examples, tok, seq_len=9, drop_last=False))
    assert len(out) == 1
    assert out[0].segment_ids.tolist() == [[0, 0, 0, 1, 1, 1, 2, 2, 2]]


def test_pack_sft_seq_len_must_be_positive() -> None:
    tok = _tok()
    with pytest.raises(ValueError, match="seq_len must be positive"):
        list(pack_sft_examples([], tok, seq_len=0))


def test_pack_sft_into_batch_groups_rows() -> None:
    tok = _tok()
    # 4 identical examples of length 5 each → 4 windows at seq_len=5.
    examples = [SFTExample(prompt="ab", response="cd")] * 4
    batches = list(pack_sft_into_batch(examples, tok, batch_size=2, seq_len=5, drop_last=True))
    assert len(batches) == 2
    for b in batches:
        assert b.tokens.shape == (2, 5)
        assert b.loss_mask.shape == (2, 5)
        assert b.segment_ids.shape == (2, 5)


def test_pack_sft_into_batch_pads_partial_when_drop_last_false() -> None:
    tok = _tok()
    # 1 example of length 5 → 1 window, batch_size=4.
    examples = [SFTExample(prompt="ab", response="cd")]
    batches = list(
        pack_sft_into_batch(examples, tok, batch_size=4, seq_len=5, drop_last=False),
    )
    assert len(batches) == 1
    b = batches[0]
    assert b.tokens.shape == (4, 5)
    # First row = real example, rows 1..3 = pad rows.
    assert b.loss_mask[0].tolist() == [0, 0, 1, 1, 1]
    assert b.loss_mask[1:].sum().item() == 0
    # Pad-row segment_ids are -1 sentinel.
    assert (b.segment_ids[1:] == -1).all().item()


def test_sft_cross_entropy_only_counts_response_positions() -> None:
    """Verify the masked-CE math: identical logits, two batches differing only
    in mask should produce different (or proportionally-related) losses.
    """
    torch.manual_seed(0)
    B, T, V = 1, 5, 8
    logits = torch.randn(B, T, V)

    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    mask_full = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
    mask_resp = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)  # last 3 are response

    # Use a fake out dict.
    out = {"logits": logits}

    full = sft_cross_entropy(out, SFTPackedBatch(tokens, mask_full, torch.zeros_like(tokens)))
    resp = sft_cross_entropy(out, SFTPackedBatch(tokens, mask_resp, torch.zeros_like(tokens)))

    # Reference: full averages over all 4 shifted positions (mask[1:] sum=4).
    # resp averages over only the last 3 shifted positions (mask[1:] sum=3 -> indices 1,2,3 of shift).
    pred = logits[:, :-1].reshape(-1, V)
    tgt = tokens[:, 1:].reshape(-1)
    per_pos = F.cross_entropy(pred, tgt, reduction="none")  # shape (T-1,) = (4,)

    expected_full = per_pos.mean()
    # mask_resp[:,1:] = [0,1,1,1] → keep positions 1,2,3 of the shifted axis.
    expected_resp = per_pos[1:].sum() / 3.0

    torch.testing.assert_close(full, expected_full)
    torch.testing.assert_close(resp, expected_resp)


def test_sft_cross_entropy_zero_mask_returns_zero_not_nan() -> None:
    """Degenerate batch with no response positions produces a safe zero."""
    B, T, V = 1, 4, 8
    logits = torch.randn(B, T, V)
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mask = torch.zeros_like(tokens)
    batch = SFTPackedBatch(tokens, mask, torch.zeros_like(tokens))

    loss = sft_cross_entropy({"logits": logits}, batch)
    assert torch.isfinite(loss).item()
    assert loss.item() == 0.0


def test_sft_cross_entropy_grad_flows_only_through_response_positions() -> None:
    """Gradient w.r.t. logits should be zero at masked-out (prompt) positions."""
    torch.manual_seed(0)
    B, T, V = 1, 6, 8
    logits = torch.randn(B, T, V, requires_grad=True)

    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    # Mask: prompt positions 0..2 (target 1..2 ignored), response 3..5.
    mask = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)
    batch = SFTPackedBatch(tokens, mask, torch.zeros_like(tokens))

    loss = sft_cross_entropy({"logits": logits}, batch)
    loss.backward()
    assert logits.grad is not None

    # CE at position t produces grad on logits[:, t, :].
    # Mask on target index t+1 gates: mask[1:] = [0,0,1,1,1] over predictor t=0..4.
    # So predictor positions 0,1 should have zero grad; 2,3,4 nonzero; 5 always zero (no target).
    grad_norm_per_pos = logits.grad.norm(dim=-1).squeeze(0)  # (T,)
    assert torch.allclose(grad_norm_per_pos[:2], torch.zeros(2))
    assert (grad_norm_per_pos[2:5] > 0).all()
    assert grad_norm_per_pos[5].item() == 0.0
