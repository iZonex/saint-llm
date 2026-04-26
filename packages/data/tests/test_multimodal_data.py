"""Tests for multimodal data plumbing."""

from __future__ import annotations

import pytest
import torch
from saint_llm_data import (
    CharTokenizer,
    MultimodalExample,
    MultimodalPackedBatch,
    TokenSlots,
    encode_multimodal,
    pack_multimodal_examples,
)


def _slots() -> TokenSlots:
    """Hand-crafted slots for CharTokenizer (no special-token registry)."""
    return TokenSlots(image_pad=4, audio_start=5, audio_end=6)


def test_encode_multimodal_text_only() -> None:
    tok = CharTokenizer()
    ex = MultimodalExample(prompt="hi", response="ok")
    ids, mask = encode_multimodal(ex, tok, _slots(), append_eos=True)
    # 2 prompt + 2 response + 1 eos = 5
    assert len(ids) == 5
    assert len(mask) == 5
    assert mask[:2] == [0, 0]
    assert mask[2:] == [1, 1, 1]


def test_encode_multimodal_with_image_features() -> None:
    tok = CharTokenizer()
    ex = MultimodalExample(
        prompt="see",
        response="ok",
        image_features=(torch.zeros(3, 16),),  # 3 patches
    )
    ids, mask = encode_multimodal(ex, tok, _slots(), append_eos=False)
    # 3 prompt chars + 3 image_pad slots + 2 response chars = 8
    assert len(ids) == 8
    # All 3 image positions should hold the image_pad ID and be loss-masked.
    assert ids[3:6] == [4, 4, 4]
    assert mask[3:6] == [0, 0, 0]
    # Response loss-active.
    assert mask[6:] == [1, 1]


def test_encode_multimodal_with_audio_features() -> None:
    tok = CharTokenizer()
    ex = MultimodalExample(
        prompt="hear",
        response="yes",
        audio_features=(torch.zeros(2, 16),),  # 2 audio tokens
    )
    ids, mask = encode_multimodal(ex, tok, _slots(), append_eos=False)
    # 4 prompt + 1 audio_start + 2 audio body + 1 audio_end + 3 response = 11
    assert len(ids) == 11
    # Audio bracket structure.
    assert ids[4] == 5  # audio_start
    assert ids[5:7] == [5, 5]  # audio body uses start id as filler
    assert ids[7] == 6  # audio_end
    # All audio positions loss-masked.
    assert mask[4:8] == [0, 0, 0, 0]
    assert mask[8:] == [1, 1, 1]


def test_encode_multimodal_mixed_modalities() -> None:
    tok = CharTokenizer()
    ex = MultimodalExample(
        prompt="x",
        response="y",
        image_features=(torch.zeros(2, 16), torch.zeros(1, 16)),  # 2 + 1 patches
        audio_features=(torch.zeros(1, 16),),  # 1 audio token
    )
    ids, _mask = encode_multimodal(ex, tok, _slots(), append_eos=False)
    # 1 prompt + 3 image_pads + 3 audio (start + body + end) + 1 response = 8
    assert len(ids) == 8
    assert ids.count(4) == 3  # image_pad x 3


def test_pack_multimodal_emits_correct_shape() -> None:
    tok = CharTokenizer()
    examples = [
        MultimodalExample(prompt="hi", response="ok"),  # 5 tokens with eos
        MultimodalExample(prompt="bye", response="cya"),  # 7 tokens with eos
    ]
    out = list(pack_multimodal_examples(
        examples, tok, _slots(), seq_len=12, drop_last=False,
    ))
    assert len(out) >= 1
    for batch in out:
        assert isinstance(batch, MultimodalPackedBatch)
        assert batch.tokens.shape == (1, 12)
        assert batch.loss_mask.shape == (1, 12)
        assert batch.segment_ids.shape == (1, 12)


def test_pack_multimodal_attaches_image_features() -> None:
    tok = CharTokenizer()
    img1 = torch.randn(2, 16)
    examples = [MultimodalExample(prompt="x", response="y", image_features=(img1,))]
    out = list(pack_multimodal_examples(
        examples, tok, _slots(), seq_len=10, drop_last=False,
    ))
    assert len(out) == 1
    assert out[0].vision_features is not None
    assert out[0].vision_features.shape == (2, 16)
    assert out[0].audio_features is None


def test_pack_multimodal_attaches_audio_features() -> None:
    tok = CharTokenizer()
    audio = torch.randn(3, 32)
    examples = [MultimodalExample(prompt="x", response="y", audio_features=(audio,))]
    out = list(pack_multimodal_examples(
        examples, tok, _slots(), seq_len=20, drop_last=False,
    ))
    assert out[0].audio_features is not None
    assert out[0].audio_features.shape == (3, 32)


def test_pack_multimodal_segment_ids_increment() -> None:
    tok = CharTokenizer()
    examples = [
        MultimodalExample(prompt="a", response="b"),
        MultimodalExample(prompt="c", response="d"),
        MultimodalExample(prompt="e", response="f"),
    ]
    out = list(pack_multimodal_examples(
        examples, tok, _slots(), seq_len=20, drop_last=False,
    ))
    seg = out[0].segment_ids[0].tolist()
    # First 3 chars belong to seg 0, next 3 to seg 1, etc.
    assert seg[0] == 0
    assert max(seg) >= 2  # at least 3 distinct examples


def test_pack_multimodal_seq_len_must_be_positive() -> None:
    tok = CharTokenizer()
    with pytest.raises(ValueError, match="seq_len must be positive"):
        list(pack_multimodal_examples([], tok, _slots(), seq_len=0))


def test_pack_multimodal_pads_residual_when_drop_last_false() -> None:
    tok = CharTokenizer()
    examples = [MultimodalExample(prompt="a", response="b")]  # 3 tokens with eos
    out = list(pack_multimodal_examples(
        examples, tok, _slots(), seq_len=10, drop_last=False,
    ))
    assert len(out) == 1
    assert out[0].tokens.shape == (1, 10)
    # Pad positions have loss_mask=0.
    assert out[0].loss_mask[0, 3:].sum().item() == 0


def test_token_slots_from_tokenizer_requires_hf() -> None:
    """CharTokenizer doesn't have the special-token registry."""
    tok = CharTokenizer()
    with pytest.raises(ValueError, match="tokenizers backend"):
        TokenSlots.from_tokenizer(tok)
