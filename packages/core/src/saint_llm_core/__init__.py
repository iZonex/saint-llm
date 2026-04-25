"""Saint LLM core: model architecture.

Modules:
    attention.csa  — Compressed Sparse Attention
    attention.hca  — Heavily Compressed Attention
    attention.swa  — Pure sliding-window MQA (first dense layers)
    residual.mhc   — Manifold-Constrained Hyper-Connections
    moe            — DeepSeekMoE with hash routing + modality bias
    mtp            — Multi-Token Prediction modules (depth-K reserved for v0.2)
    multimodal     — Vision/audio projector slots, generation head hook, side-channel slot
    model          — Top-level transformer assembly (SaintLLM)
    config         — Pydantic model config schemas
"""

from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM

__version__ = "0.0.1"
__all__ = ["ModelConfig", "SaintLLM"]
