"""Tests for the video tokenizer interface + FakeVideoTokenizer."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.multimodal.video import (
    FakeVideoTokenizer,
    encode_video_clips,
)


def test_fake_tokenizer_shape() -> None:
    tok = FakeVideoTokenizer(vision_dim=16, patches_per_frame=4)
    video = torch.randn(8, 3, 32, 32)  # 8 frames
    out = tok.encode(video)
    assert out.shape == (8 * 4, 16)


def test_fake_tokenizer_deterministic_across_instances() -> None:
    """Same seed -> same projection matrix -> same output."""
    a = FakeVideoTokenizer(vision_dim=8, patches_per_frame=2, seed=42)
    b = FakeVideoTokenizer(vision_dim=8, patches_per_frame=2, seed=42)
    video = torch.randn(2, 3, 8, 8)
    assert torch.equal(a.encode(video), b.encode(video))


def test_fake_tokenizer_different_seeds_diverge() -> None:
    a = FakeVideoTokenizer(vision_dim=8, patches_per_frame=2, seed=1)
    b = FakeVideoTokenizer(vision_dim=8, patches_per_frame=2, seed=2)
    video = torch.randn(2, 3, 8, 8)
    assert not torch.equal(a.encode(video), b.encode(video))


def test_fake_tokenizer_patches_within_frame_distinguishable() -> None:
    """The per-patch offset means different patches within a frame differ."""
    tok = FakeVideoTokenizer(vision_dim=4, patches_per_frame=3)
    video = torch.zeros(1, 3, 4, 4)  # constant input
    out = tok.encode(video)
    # 3 patches per frame, all from the same frame -> outputs should
    # differ via the per-patch offset.
    assert not torch.equal(out[0], out[1])
    assert not torch.equal(out[1], out[2])


def test_fake_tokenizer_rejects_non_4d() -> None:
    tok = FakeVideoTokenizer(vision_dim=4, patches_per_frame=2)
    with pytest.raises(ValueError, match=r"\(T, C, H, W\)"):
        tok.encode(torch.randn(3, 8, 8))


def test_fake_tokenizer_rejects_non_rgb() -> None:
    tok = FakeVideoTokenizer(vision_dim=4, patches_per_frame=2)
    with pytest.raises(ValueError, match="3 channels"):
        tok.encode(torch.randn(2, 4, 8, 8))


def test_fake_tokenizer_rejects_zero_dims() -> None:
    with pytest.raises(ValueError, match="vision_dim"):
        FakeVideoTokenizer(vision_dim=0, patches_per_frame=2)
    with pytest.raises(ValueError, match="patches_per_frame"):
        FakeVideoTokenizer(vision_dim=4, patches_per_frame=0)


def test_encode_video_clips_returns_parallel_list() -> None:
    tok = FakeVideoTokenizer(vision_dim=4, patches_per_frame=2)
    clips = [torch.randn(2, 3, 8, 8), torch.randn(3, 3, 8, 8)]
    feats = encode_video_clips(clips, tok)
    assert len(feats) == 2
    assert feats[0].shape == (4, 4)  # 2 frames * 2 patches
    assert feats[1].shape == (6, 4)  # 3 frames * 2 patches


def test_fake_tokenizer_compatible_with_multimodal_pipeline() -> None:
    """Output shape (N, vision_dim) drops into MultimodalExample.image_features."""
    tok = FakeVideoTokenizer(vision_dim=96, patches_per_frame=4)
    video = torch.randn(5, 3, 32, 32)
    feats = tok.encode(video)
    # MultimodalExample expects 2D tensors with rows=image_pad_count.
    assert feats.dim() == 2
    assert feats.shape[1] == 96


def test_vision_dim_and_patches_per_frame_properties() -> None:
    tok = FakeVideoTokenizer(vision_dim=32, patches_per_frame=8)
    assert tok.vision_dim == 32
    assert tok.patches_per_frame == 8
