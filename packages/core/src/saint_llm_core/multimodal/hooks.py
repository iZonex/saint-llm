"""Multimodal hooks reserved in v0.1.

Per AUGMENTATIONS.md:
    MM-V-02/05  Vision projector slot (LLaVA-style soft tokens at <|image_pad|>).
    MM-A-01/03  Audio projector slot (Whisper-large-v3 → 12.5Hz → MLP → d_model).
    MM-V-08     Generation head hook (zero-init unembedding for VQ codebook IDs, frozen v0.1).
    MEM-03      Residual side-channel slot per block (Titans/MIRAS escape hatch, alpha=0).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from saint_llm_core.attention.common import RMSNorm


class ModalityProjector(nn.Module):
    """2-layer SwiGLU MLP: external encoder dim → model hidden dim.

    Identity (zero-output) when disabled — keeps shape contract for unused modalities.
    """

    def __init__(self, in_dim: int, hidden_dim: int, enabled: bool = False) -> None:
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.up = nn.Linear(in_dim, hidden_dim, bias=False)
            self.gate = nn.Linear(in_dim, hidden_dim, bias=False)
            self.down = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.out_norm = RMSNorm(hidden_dim)
        else:
            self.register_parameter("_disabled", nn.Parameter(torch.zeros(1), requires_grad=False))

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled:
            raise RuntimeError("ModalityProjector is disabled — provided no input but called forward()")
        return self.out_norm(self.down(F.silu(self.gate(x)) * self.up(x)))


class ResidualSideChannel(nn.Module):
    """Per-block side channel: h <- h + alpha * MemModule(h).

    v0.1: MemModule = identity, alpha = 0 (gated off). The slot is reserved so that a v0.2
    Titans/MIRAS-style neural memory module can be attached without checkpoint surgery.
    """

    def __init__(self, hidden_dim: int, alpha_init: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))
        self._mem_module: nn.Module | None = None

    def attach(self, mem_module: nn.Module) -> None:
        """v0.2 hook — attach a real memory module."""
        self._mem_module = mem_module

    def forward(self, h: Tensor) -> Tensor:
        if self.alpha.detach().abs().item() == 0.0 or self._mem_module is None:
            return h
        return h + self.alpha * self._mem_module(h)


class GenerationHeadHook(nn.Module):
    """Reserved zero-init projection from final hidden to VQ codebook IDs (v0.2 image generation).

    In v0.1 this is a frozen zero-init linear; presence reserves the parameter rows so that the
    Janus-Pro-style VQ tokenizer can be attached in v0.2 without resizing embeddings.
    """

    def __init__(self, hidden_dim: int, vocab_slots: int, enabled: bool = False) -> None:
        super().__init__()
        self.enabled = enabled
        self.proj = nn.Linear(hidden_dim, vocab_slots, bias=False)
        nn.init.zeros_(self.proj.weight)
        if not enabled:
            for p in self.proj.parameters():
                p.requires_grad_(False)

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(h)
