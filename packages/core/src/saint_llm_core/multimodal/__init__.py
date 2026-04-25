"""Multimodal hooks (reserved in v0.1) + vision encoder integration (task #14)."""

from saint_llm_core.multimodal.hooks import (
    GenerationHeadHook,
    ModalityProjector,
    ResidualSideChannel,
)
from saint_llm_core.multimodal.vision import (
    FakeViT,
    SigLIP2Wrapper,
    VisionEncoderConfig,
    VisionTokenizer,
    deepstack_fuse,
)

__all__ = [
    "FakeViT",
    "GenerationHeadHook",
    "ModalityProjector",
    "ResidualSideChannel",
    "SigLIP2Wrapper",
    "VisionEncoderConfig",
    "VisionTokenizer",
    "deepstack_fuse",
]
