"""Saint LLM kernels.

Modules:
    mhc            — fused mHC fwd/bwd kernels
    attention      — batch-invariant deterministic attention (dual-kernel decoding)
    moe            — MegaMoE-style EP mega-kernel + deterministic backward
    matmul         — split-k mHC deterministic matmul
    quant          — FP4/FP8 cast/dequant kernels
"""

__version__ = "0.0.1"
