"""Tests for multimodal chat rendering."""

from __future__ import annotations

import pytest
import torch
from saint_llm_data import CharTokenizer
from saint_llm_data.chat_template import ChatTurn, render_chat
from saint_llm_data.multimodal import TokenSlots
from saint_llm_data.multimodal_chat import (
    MultimodalChatTurn,
    RenderedMultimodalChat,
    render_multimodal_chat,
    render_text_chat_to_multimodal,
    to_chat_turn,
)


def _slots() -> TokenSlots:
    return TokenSlots(image_pad=200, audio_start=201, audio_end=202)


def test_text_only_render_matches_chat_template() -> None:
    """Without any modalities, rendering should match render_chat byte-for-byte."""
    tok = CharTokenizer()
    text_turns = [
        ChatTurn(role="user", content="hi"),
        ChatTurn(role="assistant", content="ok"),
    ]
    mm_turns = [
        MultimodalChatTurn(role="user", content="hi"),
        MultimodalChatTurn(role="assistant", content="ok"),
    ]
    text_out = render_chat(text_turns, tok)
    mm_out = render_multimodal_chat(mm_turns, tok, _slots())
    assert mm_out.token_ids == text_out.token_ids
    assert mm_out.loss_mask == text_out.loss_mask
    assert mm_out.vision_features is None
    assert mm_out.audio_features is None


def test_image_features_splice_image_pad_tokens() -> None:
    """A user turn with image_features inserts image_pad after content."""
    tok = CharTokenizer()
    img = torch.randn(3, 16)  # 3 patches
    turns = [
        MultimodalChatTurn(role="user", content="see", image_features=(img,)),
        MultimodalChatTurn(role="assistant", content="ok"),
    ]
    out = render_multimodal_chat(turns, tok, _slots())
    assert out.token_ids.count(_slots().image_pad) == 3
    assert out.vision_features is not None
    assert out.vision_features.shape == (3, 16)
    assert torch.equal(out.vision_features, img)


def test_audio_features_bracket_correctly() -> None:
    """A user turn with audio_features brackets with audio_start/end."""
    tok = CharTokenizer()
    audio = torch.randn(2, 8)  # 2 audio tokens
    turns = [
        MultimodalChatTurn(role="user", content="hear", audio_features=(audio,)),
        MultimodalChatTurn(role="assistant", content="yes"),
    ]
    out = render_multimodal_chat(turns, tok, _slots())
    s = _slots()
    # Find the audio bracket: 1 start + 2 body + 1 end.
    audio_start_count = out.token_ids.count(s.audio_start)
    audio_end_count = out.token_ids.count(s.audio_end)
    assert audio_start_count == 1 + 2  # start + body uses start ID
    assert audio_end_count == 1
    # Audio features include zero-padded ends so flat tensor lines up
    # with every slot in [start, end]: 1 + 2 + 1 = 4 rows.
    assert out.audio_features is not None
    assert out.audio_features.shape == (4, 8)


def test_multiple_images_concatenated_in_render_order() -> None:
    """Two images in one turn -> features stacked in order."""
    tok = CharTokenizer()
    img1 = torch.randn(2, 16)
    img2 = torch.randn(3, 16)
    turns = [
        MultimodalChatTurn(
            role="user", content="x", image_features=(img1, img2),
        ),
        MultimodalChatTurn(role="assistant", content="ok"),
    ]
    out = render_multimodal_chat(turns, tok, _slots())
    assert out.vision_features is not None
    assert out.vision_features.shape == (5, 16)
    assert torch.equal(out.vision_features[:2], img1)
    assert torch.equal(out.vision_features[2:], img2)


def test_render_loss_mask_zero_on_image_pad_positions() -> None:
    """Image pad tokens are loss-masked (model uses splice not LM head there)."""
    tok = CharTokenizer()
    img = torch.randn(2, 16)
    turns = [
        MultimodalChatTurn(role="user", content="x", image_features=(img,)),
        MultimodalChatTurn(role="assistant", content="ok"),
    ]
    out = render_multimodal_chat(turns, tok, _slots())
    s = _slots()
    for tok_id, mask in zip(out.token_ids, out.loss_mask, strict=True):
        if tok_id == s.image_pad:
            assert mask == 0


def test_render_assistant_content_loss_active() -> None:
    tok = CharTokenizer()
    turns = [
        MultimodalChatTurn(role="user", content="hi"),
        MultimodalChatTurn(role="assistant", content="hello"),
    ]
    out = render_multimodal_chat(turns, tok, _slots())
    pos = out.text.find("hello")
    for i in range(len("hello")):
        assert out.loss_mask[pos + i] == 1


def test_render_unknown_role_raises() -> None:
    tok = CharTokenizer()
    with pytest.raises(ValueError, match="unknown role"):
        render_multimodal_chat(
            [MultimodalChatTurn(role="bogus", content="x")],
            tok, _slots(),
        )


def test_render_rejects_non_2d_image_features() -> None:
    tok = CharTokenizer()
    with pytest.raises(ValueError, match=r"\(n_patches, vision_dim\)"):
        render_multimodal_chat(
            [MultimodalChatTurn(
                role="user", content="x",
                image_features=(torch.randn(3, 4, 4),),  # 3D — bad
            )],
            tok, _slots(),
        )


def test_render_rejects_non_2d_audio_features() -> None:
    tok = CharTokenizer()
    with pytest.raises(ValueError, match=r"\(n_tokens, audio_dim\)"):
        render_multimodal_chat(
            [MultimodalChatTurn(
                role="user", content="x",
                audio_features=(torch.randn(5),),  # 1D — bad
            )],
            tok, _slots(),
        )


def test_render_with_generation_prompt() -> None:
    tok = CharTokenizer()
    turns = [MultimodalChatTurn(role="user", content="hi")]
    out = render_multimodal_chat(turns, tok, _slots(), add_generation_prompt=True)
    assert out.text.endswith("<|assistant|>\n")


def test_render_to_torch_returns_2d_tensors() -> None:
    tok = CharTokenizer()
    turns = [MultimodalChatTurn(role="user", content="hi")]
    out = render_multimodal_chat(turns, tok, _slots())
    toks, mask = out.to_torch()
    assert toks.shape == (1, len(out.token_ids))
    assert mask.shape == toks.shape


def test_to_chat_turn_strips_modalities() -> None:
    img = torch.randn(2, 8)
    mm = MultimodalChatTurn(
        role="user", content="hi", image_features=(img,),
    )
    text = to_chat_turn(mm)
    assert isinstance(text, ChatTurn)
    assert text.role == "user"
    assert text.content == "hi"


def test_render_text_chat_to_multimodal_passes_through() -> None:
    tok = CharTokenizer()
    text_out = render_chat(
        [ChatTurn(role="user", content="hi")], tok,
    )
    mm_out = render_text_chat_to_multimodal(text_out)
    assert isinstance(mm_out, RenderedMultimodalChat)
    assert mm_out.token_ids == text_out.token_ids
    assert mm_out.vision_features is None
    assert mm_out.audio_features is None


def test_render_multi_turn_with_one_image_per_user_turn() -> None:
    """Two user turns each with their own image -> features concatenated."""
    tok = CharTokenizer()
    img1 = torch.randn(2, 16)
    img2 = torch.randn(1, 16)
    turns = [
        MultimodalChatTurn(role="user", content="first", image_features=(img1,)),
        MultimodalChatTurn(role="assistant", content="ok"),
        MultimodalChatTurn(role="user", content="second", image_features=(img2,)),
        MultimodalChatTurn(role="assistant", content="ok"),
    ]
    out = render_multimodal_chat(turns, tok, _slots())
    assert out.vision_features is not None
    assert out.vision_features.shape == (3, 16)
    # Render order: img1 (2 rows) then img2 (1 row).
    assert torch.equal(out.vision_features[:2], img1)
    assert torch.equal(out.vision_features[2:3], img2)


def test_render_image_count_matches_image_pad_count() -> None:
    """Invariant: feature row count == # of image_pad tokens emitted."""
    tok = CharTokenizer()
    img1 = torch.randn(4, 32)
    img2 = torch.randn(7, 32)
    turns = [
        MultimodalChatTurn(
            role="user", content="x", image_features=(img1, img2),
        ),
        MultimodalChatTurn(role="assistant", content="ok"),
    ]
    out = render_multimodal_chat(turns, tok, _slots())
    n_image_pad = out.token_ids.count(_slots().image_pad)
    assert out.vision_features is not None
    assert out.vision_features.shape[0] == n_image_pad == 11
