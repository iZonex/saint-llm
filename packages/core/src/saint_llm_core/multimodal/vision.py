"""Vision encoder integration: SigLIP-2-SO400M wrapper + DeepStack feature fusion.

Per AUGMENTATIONS.md:
    MM-V-02  SigLIP-2-SO400M ViT encoder
    MM-V-04  DeepStack multi-level feature fusion (concat features from layers ~½, ¾, final)
    MM-V-05  Adaptive token budget 64-1280 visual tokens per image
    MM-V-07  Frozen in v0.1, plan stage-2 unfreeze

The encoder produces (B, n_patches, vision_dim) features that the model's `vision_proj`
consumes. Caller flattens to (n_image_tokens_total, vision_dim) before passing to
`SaintLLM.forward(..., vision_features=...)`.

For unit testing without HF/internet we expose `FakeViT`. For real validation use
`SigLIP2Wrapper.from_pretrained(...)` on a CUDA box (lazy HF load).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Sequence


class VisionEncoderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    encoder_name: str = "google/siglip2-so400m-patch14-384"
    encoder_hidden_dim: int = 1152
    deepstack_layers: tuple[int, ...] = (-1, -5, -9)
    deepstack_enabled: bool = True
    freeze: bool = True
    image_size: int = 384
    patch_size: int = 14
    adaptive_token_min: int = 64
    adaptive_token_max: int = 1280

    @property
    def output_dim(self) -> int:
        if self.deepstack_enabled:
            return self.encoder_hidden_dim * len(self.deepstack_layers)
        return self.encoder_hidden_dim


def deepstack_fuse(hidden_states: "Sequence[Tensor]", layers: tuple[int, ...]) -> Tensor:
    """Concatenate per-patch features from selected ViT layers along the hidden dim.

    hidden_states: list of (B, n_patches, encoder_dim) — one per ViT layer (incl embeddings).
    layers: indices into hidden_states (negatives allowed).
    Returns (B, n_patches, len(layers) * encoder_dim).
    """
    selected = [hidden_states[i] for i in layers]
    return torch.cat(selected, dim=-1)


class FakeViT(nn.Module):
    """Zero-dependency stand-in for unit tests.

    Mimics the SigLIP-2-SO400M output contract: input (B, 3, H, W) → hidden_states list
    of (B, n_patches, encoder_dim) per layer. Uses random projections so the test exercises
    DeepStack fusion + projector + token-replacement plumbing without downloading anything.
    """

    def __init__(self, cfg: VisionEncoderConfig, n_layers: int = 12) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_layers = n_layers
        self.patch_proj = nn.Conv2d(3, cfg.encoder_hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.layer_perturbations = nn.ParameterList(
            nn.Parameter(torch.randn(cfg.encoder_hidden_dim) * 0.01) for _ in range(n_layers)
        )

    def forward(self, pixel_values: Tensor) -> dict[str, Tensor | list[Tensor]]:
        b = pixel_values.shape[0]
        # (B, encoder_hidden_dim, H/p, W/p) → (B, n_patches, encoder_hidden_dim)
        patches = self.patch_proj(pixel_values).flatten(2).transpose(1, 2)
        n_patches = patches.shape[1]

        hidden_states: list[Tensor] = [patches]
        cur = patches
        for perturb in self.layer_perturbations:
            cur = cur + perturb.view(1, 1, -1)
            hidden_states.append(cur)

        return {
            "last_hidden_state": hidden_states[-1],
            "hidden_states": hidden_states,
            "n_patches": torch.tensor(n_patches),
        }


class SigLIP2Wrapper(nn.Module):
    """Real SigLIP-2-SO400M wrapper. Lazy-loads from HuggingFace on first forward.

    Use only in environments with HF Hub access + enough disk for ~1.5GB weights.
    For unit tests use `FakeViT`. For integration tests use `@pytest.mark.gpu @pytest.mark.slow`.
    """

    def __init__(self, cfg: VisionEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._encoder: nn.Module | None = None

    def _ensure_loaded(self) -> nn.Module:
        if self._encoder is not None:
            return self._encoder
        # Local import — keeps `transformers` an optional runtime dep for the core package.
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(self.cfg.encoder_name)
        # SigLIP returns SiglipModel; we want only the vision part.
        if hasattr(encoder, "vision_model"):
            encoder = encoder.vision_model
        if self.cfg.freeze:
            for p in encoder.parameters():
                p.requires_grad_(False)
            encoder.eval()
        self._encoder = encoder
        return encoder

    def forward(self, pixel_values: Tensor) -> dict[str, Tensor | list[Tensor]]:
        encoder = self._ensure_loaded()
        outputs = encoder(pixel_values, output_hidden_states=True)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "hidden_states": list(outputs.hidden_states),
        }


class VisionTokenizer(nn.Module):
    """Wrap any encoder + DeepStack fusion. Produces final per-patch features the LM consumes.

    Output: (B, n_patches, output_dim) where output_dim = cfg.output_dim.
    """

    def __init__(self, cfg: VisionEncoderConfig, encoder: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder

    def forward(self, pixel_values: Tensor) -> Tensor:
        outputs = self.encoder(pixel_values)
        if self.cfg.deepstack_enabled:
            return deepstack_fuse(outputs["hidden_states"], self.cfg.deepstack_layers)
        return outputs["last_hidden_state"]
