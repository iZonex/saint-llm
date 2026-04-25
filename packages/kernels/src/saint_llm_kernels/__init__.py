"""Saint LLM kernels.

Modules:
    mhc            — fused mHC fwd/bwd kernels
    attention      — batch-invariant deterministic attention (dual-kernel decoding)
    moe            — MegaMoE-style EP mega-kernel + deterministic backward
    matmul         — split-k mHC deterministic matmul
    quant          — FP4/FP8 cast/dequant kernels
"""

from saint_llm_kernels.attention import (
    lightning_indexer_scores,
    lightning_indexer_scores_reference,
    lightning_indexer_topk,
    lightning_indexer_topk_reference,
)
from saint_llm_kernels.fp8_gemm import fp8_gemm, is_fp8_gemm_supported
from saint_llm_kernels.linear_fp4 import Fp4Linear
from saint_llm_kernels.linear_fp8 import Fp8Linear
from saint_llm_kernels.mhc import mhc_carry, mhc_carry_reference
from saint_llm_kernels.moe_grouped import GroupedSwiGLUExperts, grouped_mm
from saint_llm_kernels.quant import (
    Fp8Format,
    cast_to_fp4_mx,
    cast_to_fp8,
    dequant_from_fp4_mx,
    dequant_from_fp8,
    fake_quant_fp4_mx,
    fake_quant_fp8,
)

__version__ = "0.0.1"

__all__ = [
    "Fp4Linear",
    "Fp8Format",
    "Fp8Linear",
    "GroupedSwiGLUExperts",
    "cast_to_fp4_mx",
    "cast_to_fp8",
    "dequant_from_fp4_mx",
    "dequant_from_fp8",
    "fake_quant_fp4_mx",
    "fake_quant_fp8",
    "fp8_gemm",
    "grouped_mm",
    "is_fp8_gemm_supported",
    "lightning_indexer_scores",
    "lightning_indexer_scores_reference",
    "lightning_indexer_topk",
    "lightning_indexer_topk_reference",
    "mhc_carry",
    "mhc_carry_reference",
]
