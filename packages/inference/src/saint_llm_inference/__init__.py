"""Saint LLM inference framework.

Modules:
    generate              — Greedy + top-k sampling decoding loops (no KV cache yet)
    kv_cache.layout       — Heterogeneous KV cache (CSA/HCA blocks of lcm(m,m'))
    kv_cache.state        — State cache for SWA + uncompressed CSA/HCA tail tokens
    kv_cache.on_disk      — Three SWA strategies (Full / Periodic / Zero)
    sparse_attn.kernel    — Sparse-attention kernel co-design with cache layout
    storage.mixed_prec    — BF16 RoPE dims + FP8 main dims
    quick_instruction     — Auxiliary-task tokens reusing precomputed KV cache
    engine                — Top-level inference engine
"""

from saint_llm_inference.generate import greedy_decode, top_k_sample

__version__ = "0.0.1"

__all__ = ["greedy_decode", "top_k_sample"]
