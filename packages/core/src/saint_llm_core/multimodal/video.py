"""Video tokenizer interface — Cosmos-Tokenizer compatible Protocol.

NVIDIA's Cosmos-Tokenizer (Jan 2025) tokenizes video clips into a
(T_temporal, H_patch, W_patch) grid of discrete or continuous tokens
ready for an LM. The real package is heavy (depends on TensorRT and
NVIDIA-specific CUDA kernels), so saint-llm v0.0 ships an *interface*
plus a deterministic ``FakeVideoTokenizer`` so:

* multimodal pipeline code can carry video features end-to-end (same
  ``image_pad`` slot machinery — video tokens splice in identically)
* tests don't block on the heavy NVIDIA dep
* production swaps in a real Cosmos wrapper by satisfying the
  :class:`VideoTokenizer` Protocol

The Protocol is intentionally minimal:

    encode(video: Tensor (T, C, H, W)) -> Tensor (N_patches, vision_dim)

Each frame contributes a fixed number of patch tokens; the temporal
axis is collapsed into the patch dimension so downstream code treats
video as "a long sequence of image patches."

Reference:
    NVIDIA Cosmos-Tokenizer (Jan 2025) — https://github.com/NVIDIA/Cosmos-Tokenizer
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn


class VideoTokenizer(Protocol):
    """Protocol every video tokenizer satisfies.

    Implementations:
        :class:`FakeVideoTokenizer` — deterministic, dep-free; for tests.
        ``CosmosVideoTokenizer``        — real NVIDIA Cosmos wrapper (v0.1).

    The output is shaped to drop directly into
    :class:`saint_llm_data.MultimodalExample.image_features` — every
    video clip becomes a single ``(N_patches, vision_dim)`` tensor.
    """

    @property
    def vision_dim(self) -> int: ...

    @property
    def patches_per_frame(self) -> int: ...

    def encode(self, video: Tensor) -> Tensor: ...


class FakeVideoTokenizer(nn.Module):
    """Deterministic, dep-free video tokenizer.

    Encodes ``(T, C, H, W)`` into ``(T * patches_per_frame, vision_dim)``
    via a fixed linear projection of mean-pooled patches. Reproducible
    given the same input — useful for unit tests + integration smokes.

    Args:
        vision_dim:        output feature dim (matches the model's
            vision projector input dim).
        patches_per_frame: how many patch tokens each frame contributes.
        seed:              torch.Generator seed for the random
            projection. Default 0 so two instances produce identical
            features.
    """

    def __init__(
        self,
        *,
        vision_dim: int,
        patches_per_frame: int,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if vision_dim <= 0:
            raise ValueError(f"vision_dim must be positive; got {vision_dim}")
        if patches_per_frame <= 0:
            raise ValueError(
                f"patches_per_frame must be positive; got {patches_per_frame}",
            )
        self._vision_dim = vision_dim
        self._patches_per_frame = patches_per_frame
        # Fixed random projection — chunk_dim -> vision_dim — so the
        # tokenizer is deterministic but covers a meaningful subspace.
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "projection",
            torch.randn(3, vision_dim, generator=gen),
            persistent=False,
        )

    @property
    def vision_dim(self) -> int:
        return self._vision_dim

    @property
    def patches_per_frame(self) -> int:
        return self._patches_per_frame

    def encode(self, video: Tensor) -> Tensor:
        """Encode ``(T, C, H, W)`` -> ``(T * patches_per_frame, vision_dim)``.

        Per frame we tile the patch axis: each frame's mean-pooled
        per-channel value gets repeated ``patches_per_frame`` times,
        with a per-patch noise term so different patches within the
        same frame are distinguishable. The values themselves are not
        meaningful — this is a stand-in for a real tokenizer.
        """
        if video.dim() != 4:
            raise ValueError(
                f"video must be (T, C, H, W); got shape {tuple(video.shape)}",
            )
        t, c, _, _ = video.shape
        if c != 3:
            raise ValueError(
                f"video must have 3 channels (RGB); got {c}",
            )
        # Mean-pool spatial dims -> (T, C).
        per_frame = video.float().mean(dim=(-2, -1))
        # Project each frame's RGB mean to vision_dim.
        projected = per_frame @ self.projection  # (T, vision_dim)
        # Tile along the patch dim with deterministic per-patch offset.
        offsets = torch.arange(
            self._patches_per_frame, device=video.device, dtype=projected.dtype,
        ).unsqueeze(-1) * 1e-3  # (P, 1)
        # Result: (T, P, vision_dim) -> (T*P, vision_dim).
        out = projected.unsqueeze(1) + offsets.unsqueeze(0)
        return out.reshape(t * self._patches_per_frame, self._vision_dim)


def encode_video_clips(
    clips: list[Tensor], tokenizer: VideoTokenizer,
) -> list[Tensor]:
    """Encode a list of clips, one per call. Returns a parallel list of features."""
    return [tokenizer.encode(clip) for clip in clips]
