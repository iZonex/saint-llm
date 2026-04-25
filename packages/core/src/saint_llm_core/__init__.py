"""Saint LLM core: model architecture.

Modules:
    attention.csa  — Compressed Sparse Attention
    attention.hca  — Heavily Compressed Attention
    residual.mhc   — Manifold-Constrained Hyper-Connections
    moe            — DeepSeekMoE with hash routing
    mtp            — Multi-Token Prediction modules
    model          — Top-level transformer assembly
    config         — Pydantic model config schemas
"""

__version__ = "0.0.1"
