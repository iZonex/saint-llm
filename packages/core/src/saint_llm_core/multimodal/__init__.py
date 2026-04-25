"""Multimodal hooks (reserved in v0.1) + vision (#14) + audio (#15) encoder integration."""

from saint_llm_core.multimodal.audio import (
    AudioEncoderConfig,
    AudioTokenizer,
    FakeWhisperEncoder,
    WhisperLargeV3Wrapper,
)
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
    "AudioEncoderConfig",
    "AudioTokenizer",
    "FakeViT",
    "FakeWhisperEncoder",
    "GenerationHeadHook",
    "ModalityProjector",
    "ResidualSideChannel",
    "SigLIP2Wrapper",
    "VisionEncoderConfig",
    "VisionTokenizer",
    "WhisperLargeV3Wrapper",
    "deepstack_fuse",
]
