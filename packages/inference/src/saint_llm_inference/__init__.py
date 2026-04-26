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

from saint_llm_inference.eagle3 import (
    EAGLE3Config,
    EAGLE3DraftHead,
    speculative_decode,
)
from saint_llm_inference.engram import (
    EngramConfig,
    EngramHook,
    EngramTable,
)
from saint_llm_inference.generate import (
    greedy_decode,
    greedy_decode_cached,
    top_k_sample,
    top_k_sample_cached,
    top_p_sample,
    top_p_sample_cached,
)
from saint_llm_inference.kv_cache import (
    CSAKVCacheLayer,
    HCAKVCacheLayer,
    KVCacheBundle,
    SWAKVCacheLayer,
)
from saint_llm_inference.multimodal_generate import (
    MultimodalSlots,
    build_multimodal_prompt_ids,
    multimodal_greedy_decode,
)
from saint_llm_inference.chat_session import ChatSession, GenerationConfig
from saint_llm_inference.multimodal_chat_session import MultimodalChatSession
from saint_llm_inference.logit_bias import (
    apply_logit_bias,
    forbid_tokens_bias,
    force_token_bias,
)
from saint_llm_inference.stop_sequences import (
    StopSequenceMatcher,
    encode_stop_strings,
)
from saint_llm_inference.streaming import (
    stream_greedy_decode,
    stream_top_p_sample,
)

__version__ = "0.0.1"

__all__ = [
    "CSAKVCacheLayer",
    "ChatSession",
    "EAGLE3Config",
    "EAGLE3DraftHead",
    "EngramConfig",
    "EngramHook",
    "EngramTable",
    "GenerationConfig",
    "HCAKVCacheLayer",
    "KVCacheBundle",
    "MultimodalChatSession",
    "MultimodalSlots",
    "SWAKVCacheLayer",
    "StopSequenceMatcher",
    "apply_logit_bias",
    "build_multimodal_prompt_ids",
    "encode_stop_strings",
    "forbid_tokens_bias",
    "force_token_bias",
    "greedy_decode",
    "greedy_decode_cached",
    "multimodal_greedy_decode",
    "speculative_decode",
    "stream_greedy_decode",
    "stream_top_p_sample",
    "top_k_sample",
    "top_k_sample_cached",
    "top_p_sample",
    "top_p_sample_cached",
]
