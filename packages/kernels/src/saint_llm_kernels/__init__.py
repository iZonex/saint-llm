"""Saint LLM kernels.

Modules:
    mhc            — fused mHC fwd/bwd kernels
    attention      — batch-invariant deterministic attention (dual-kernel decoding)
    moe            — MegaMoE-style EP mega-kernel + deterministic backward
    matmul         — split-k mHC deterministic matmul
    quant          — FP4/FP8 cast/dequant kernels
"""

from saint_llm_kernels.quant import (
    Fp8Format,
    cast_to_fp8,
    dequant_from_fp8,
    fake_quant_fp8,
)

__version__ = "0.0.1"

__all__ = [
    "Fp8Format",
    "cast_to_fp8",
    "dequant_from_fp8",
    "fake_quant_fp8",
]
