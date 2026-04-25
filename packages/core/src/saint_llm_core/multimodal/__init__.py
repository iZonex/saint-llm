"""Multimodal hooks reserved in v0.1 — actual encoders attached in tasks #14 (vision) and #15 (audio)."""

from saint_llm_core.multimodal.hooks import (
    GenerationHeadHook,
    ModalityProjector,
    ResidualSideChannel,
)

__all__ = ["GenerationHeadHook", "ModalityProjector", "ResidualSideChannel"]
