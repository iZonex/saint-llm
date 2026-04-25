"""SWAKVCacheLayer + SWAttention cached forward equivalence."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.attention import SWAttention
from saint_llm_core.config import ModelConfig
from saint_llm_inference import SWAKVCacheLayer


def _swa(cfg: ModelConfig) -> SWAttention:
    torch.manual_seed(0)
    layer = SWAttention(cfg.hidden_dim, cfg.attention)
    layer.eval()
    return layer


# ---------- SWAKVCacheLayer ----------


def test_cache_initial_state_empty() -> None:
    cache = SWAKVCacheLayer(max_seq_len=16, head_dim=32)
    assert cache.length == 0
    assert cache.k.shape == (1, 16, 32)
    assert cache.v.shape == (1, 16, 32)


def test_cache_append_advances_length() -> None:
    cache = SWAKVCacheLayer(max_seq_len=16, head_dim=8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    full_k, full_v = cache.append(k, v)
    assert cache.length == 4
    assert full_k.shape == (1, 4, 8)
    assert torch.equal(full_k, k)
    assert torch.equal(full_v, v)


def test_cache_multiple_appends_concatenate() -> None:
    cache = SWAKVCacheLayer(max_seq_len=16, head_dim=8)
    a_k = torch.randn(1, 2, 8)
    a_v = torch.randn(1, 2, 8)
    b_k = torch.randn(1, 3, 8)
    b_v = torch.randn(1, 3, 8)
    cache.append(a_k, a_v)
    full_k, full_v = cache.append(b_k, b_v)
    assert cache.length == 5
    assert torch.equal(full_k[:, :2], a_k)
    assert torch.equal(full_k[:, 2:5], b_k)
    assert torch.equal(full_v[:, :2], a_v)
    assert torch.equal(full_v[:, 2:5], b_v)


def test_cache_overflow_raises() -> None:
    cache = SWAKVCacheLayer(max_seq_len=4, head_dim=8)
    k = torch.randn(1, 5, 8)
    v = torch.randn(1, 5, 8)
    with pytest.raises(RuntimeError, match="overflow"):
        cache.append(k, v)


def test_cache_reset_clears_length() -> None:
    cache = SWAKVCacheLayer(max_seq_len=16, head_dim=8)
    cache.append(torch.randn(1, 4, 8), torch.randn(1, 4, 8))
    cache.reset()
    assert cache.length == 0


def test_cache_invalid_max_seq_len() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        SWAKVCacheLayer(max_seq_len=0, head_dim=8)


def test_cache_rejects_non_3d_inputs() -> None:
    cache = SWAKVCacheLayer(max_seq_len=4, head_dim=8)
    with pytest.raises(ValueError, match="must be 3D"):
        cache.append(torch.randn(1, 8), torch.randn(1, 8))


# ---------- SWAttention with KV cache ----------


def test_swa_cached_full_pass_matches_uncached() -> None:
    """Feed a single (1, T) batch through SWAttention twice — once without
    cache, once token-by-token with cache — outputs must be allclose."""
    cfg = ModelConfig.tiny()
    layer = _swa(cfg)
    t = 8
    h = torch.randn(1, t, cfg.hidden_dim)

    with torch.no_grad():
        out_full = layer(h)

    cache = SWAKVCacheLayer(
        max_seq_len=t,
        head_dim=cfg.attention.head_dim,
    )
    out_chunks = []
    with torch.no_grad():
        for i in range(t):
            chunk = layer(h[:, i:i + 1, :], kv_cache=cache)
            out_chunks.append(chunk)
    out_cached = torch.cat(out_chunks, dim=1)
    assert out_cached.shape == out_full.shape
    assert torch.allclose(out_cached, out_full, atol=1.0e-4)


def test_swa_cached_prefill_then_decode() -> None:
    """Realistic generate: prefill K tokens at once, then decode 1 at a time."""
    cfg = ModelConfig.tiny()
    layer = _swa(cfg)
    prompt_t = 6
    decode_t = 4
    full_t = prompt_t + decode_t
    h_full = torch.randn(1, full_t, cfg.hidden_dim)

    with torch.no_grad():
        out_full = layer(h_full)

    cache = SWAKVCacheLayer(
        max_seq_len=full_t,
        head_dim=cfg.attention.head_dim,
    )
    with torch.no_grad():
        prefill_out = layer(h_full[:, :prompt_t, :], kv_cache=cache)
        decode_chunks = [
            prefill_out,
            *(layer(h_full[:, i:i + 1, :], kv_cache=cache) for i in range(prompt_t, full_t)),
        ]
    out_cached = torch.cat(decode_chunks, dim=1)
    assert torch.allclose(out_cached, out_full, atol=1.0e-4)


def test_swa_no_cache_unchanged() -> None:
    """Calling with ``kv_cache=None`` (the default) must be byte-identical to
    the pre-cache implementation."""
    cfg = ModelConfig.tiny()
    layer = _swa(cfg)
    h = torch.randn(1, 8, cfg.hidden_dim)
    with torch.no_grad():
        a = layer(h)
        b = layer(h, kv_cache=None)
    assert torch.equal(a, b)
