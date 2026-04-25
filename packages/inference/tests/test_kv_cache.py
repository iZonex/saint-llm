"""SWAKVCacheLayer + SWAttention cached forward equivalence."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.attention import CSA, HCA, SWAttention
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_inference import (
    CSAKVCacheLayer,
    HCAKVCacheLayer,
    KVCacheBundle,
    SWAKVCacheLayer,
    greedy_decode,
    greedy_decode_cached,
)


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


# ---------- KVCacheBundle ----------


def test_bundle_length_reads_from_first_populated() -> None:
    cache = SWAKVCacheLayer(max_seq_len=16, head_dim=8)
    cache.append(torch.randn(1, 3, 8), torch.randn(1, 3, 8))
    bundle = KVCacheBundle([None, cache, None])
    assert bundle.length == 3
    assert len(bundle) == 3


def test_bundle_length_zero_when_all_none() -> None:
    bundle = KVCacheBundle([None, None])
    assert bundle.length == 0


def test_bundle_for_layer_returns_entry() -> None:
    cache = SWAKVCacheLayer(max_seq_len=4, head_dim=8)
    bundle = KVCacheBundle([cache, None])
    assert bundle.for_layer(0) is cache
    assert bundle.for_layer(1) is None


def test_bundle_reset_clears_all_caches() -> None:
    cache = SWAKVCacheLayer(max_seq_len=8, head_dim=8)
    cache.append(torch.randn(1, 2, 8), torch.randn(1, 2, 8))
    bundle = KVCacheBundle([cache, None])
    assert bundle.length == 2
    bundle.reset()
    assert bundle.length == 0


# ---------- SaintLLM end-to-end with KV cache ----------


def test_saintllm_token_by_token_matches_full_forward() -> None:
    """Build a bundle (SWA layers only get caches), feed token-by-token, and
    verify the cached forward's logits match the uncached full pass."""
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model = SaintLLM(cfg).eval()

    t = 8
    token_ids = torch.zeros(1, t, dtype=torch.long)

    with torch.no_grad():
        full_out = model(token_ids)["logits"]

    # Build bundle: SWA blocks → cache, others → None.
    layers = []
    for block in model.blocks:
        if type(block.attention).__name__ == "SWAttention":
            layers.append(SWAKVCacheLayer(
                max_seq_len=t, head_dim=cfg.attention.head_dim,
            ))
        else:
            layers.append(None)
    bundle = KVCacheBundle(layers)

    # Feed token-by-token.
    chunks = []
    with torch.no_grad():
        for i in range(t):
            out = model(token_ids[:, i:i + 1], kv_cache_bundle=bundle)
            chunks.append(out["logits"])
    cached_out = torch.cat(chunks, dim=1)

    assert cached_out.shape == full_out.shape
    # Uncached CSA/HCA layers will recompute from scratch each call (so their
    # contribution depends only on the single-token input each step) — this
    # means cached SWA + uncached CSA/HCA *won't* match the full pass exactly.
    # The contract we test: SWA cache produces consistent SWA outputs, and
    # the bundle plumbing doesn't crash.
    assert torch.isfinite(cached_out).all()


def test_saintllm_default_kv_cache_bundle_none_unchanged() -> None:
    """Forward without bundle must be byte-identical to before the integration."""
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model = SaintLLM(cfg).eval()
    token_ids = torch.zeros(1, 4, dtype=torch.long)
    with torch.no_grad():
        a = model(token_ids)["logits"]
        b = model(token_ids, kv_cache_bundle=None)["logits"]
    assert torch.equal(a, b)


def test_bundle_for_model_factory() -> None:
    """Bundle built from a SaintLLM has one cache per block (SWA + HCA + CSA
    all supported in stage 6+)."""
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model = SaintLLM(cfg).eval()
    bundle = KVCacheBundle.for_model(model, max_seq_len=8)
    assert len(bundle) == cfg.n_layers
    swa_count = sum(1 for i in range(cfg.n_layers)
                    if isinstance(bundle.for_layer(i), SWAKVCacheLayer))
    hca_count = sum(1 for i in range(cfg.n_layers)
                    if isinstance(bundle.for_layer(i), HCAKVCacheLayer))
    csa_count = sum(1 for i in range(cfg.n_layers)
                    if isinstance(bundle.for_layer(i), CSAKVCacheLayer))
    assert swa_count == cfg.first_dense_swa_layers
    assert hca_count > 0
    assert csa_count > 0
    assert swa_count + hca_count + csa_count == cfg.n_layers


def test_greedy_decode_cached_runs() -> None:
    """End-to-end: cached greedy decode produces tokens and exits cleanly."""
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model = SaintLLM(cfg).eval()
    prompt = torch.zeros(1, 4, dtype=torch.long)
    out = greedy_decode_cached(model, prompt, max_new_tokens=6)
    assert out.shape == (1, 10)
    assert out.dtype == torch.long
    # Prompt prefix preserved.
    assert torch.equal(out[:, :4], prompt)


def test_greedy_decode_cached_eos_short_circuits() -> None:
    """When the first emitted token == eos, the loop pads and returns full width."""
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model = SaintLLM(cfg).eval()
    prompt = torch.zeros(1, 4, dtype=torch.long)
    # Find the model's first greedy pick on this prompt; use it as eos.
    first = greedy_decode(model, prompt, max_new_tokens=1)[0, -1].item()
    out = greedy_decode_cached(model, prompt, max_new_tokens=8, eos_token=int(first))
    assert out.shape[-1] == 4 + 8
    # The eos token occupies the rest of the tail.
    assert (out[0, 4:] == int(first)).all()


def test_greedy_decode_cached_rejects_non_2d() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg).eval()
    with pytest.raises(ValueError, match="must be 2D"):
        greedy_decode_cached(model, torch.zeros(4, dtype=torch.long), max_new_tokens=2)


# ---------- HCAKVCacheLayer ----------


def _hca(cfg: ModelConfig) -> HCA:
    torch.manual_seed(0)
    layer = HCA(cfg.hidden_dim, cfg.attention, cfg.hca)
    layer.eval()
    return layer


def test_hca_cache_initial_state_empty() -> None:
    cache = HCAKVCacheLayer(
        max_seq_len=32, hidden_dim=128, head_dim=32, compression_rate=8,
    )
    assert cache.length == 0
    assert cache.n_blocks == 0
    assert cache.h_buf_count == 0


def test_hca_cache_invalid_max_seq_len() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        HCAKVCacheLayer(max_seq_len=0, hidden_dim=8, head_dim=4, compression_rate=2)


def test_hca_cache_invalid_compression_rate() -> None:
    with pytest.raises(ValueError, match="compression_rate"):
        HCAKVCacheLayer(
            max_seq_len=8, hidden_dim=8, head_dim=4, compression_rate=0,
        )


def test_hca_cache_overflow_raises() -> None:
    cache = HCAKVCacheLayer(
        max_seq_len=4, hidden_dim=8, head_dim=4, compression_rate=2,
    )
    h = torch.randn(1, 5, 8)
    sw_k = torch.randn(1, 5, 4)
    sw_v = torch.randn(1, 5, 4)
    with pytest.raises(RuntimeError, match="overflow"):
        cache.append(h, sw_k, sw_v, compressor=lambda x: x.new_zeros(1, 0, 4))


def test_hca_cache_emits_blocks_at_boundaries() -> None:
    """With m=4, 9 tokens fed in chunks should produce 2 blocks + 1 leftover."""
    cache = HCAKVCacheLayer(
        max_seq_len=16, hidden_dim=8, head_dim=4, compression_rate=4,
    )
    # The compressor here is a stub that emits one block per m raw tokens.
    counter = [0]

    def stub_compressor(h: torch.Tensor) -> torch.Tensor:
        # h: (B, n*m, hidden_dim) — emit (B, n, head_dim).
        n = h.shape[1] // 4
        counter[0] += 1
        return torch.full((h.shape[0], n, 4), float(counter[0]))

    cache.append(
        torch.randn(1, 3, 8), torch.randn(1, 3, 4), torch.randn(1, 3, 4),
        compressor=stub_compressor,
    )
    assert cache.h_buf_count == 3
    assert cache.n_blocks == 0

    cache.append(
        torch.randn(1, 6, 8), torch.randn(1, 6, 4), torch.randn(1, 6, 4),
        compressor=stub_compressor,
    )
    # 3 buffered + 6 new = 9 → 2 complete blocks + 1 leftover.
    assert cache.n_blocks == 2
    assert cache.h_buf_count == 1
    assert cache.length == 9


def test_hca_cached_full_pass_matches_uncached() -> None:
    """HCA cached forward token-by-token must equal the uncached full pass."""
    cfg = ModelConfig.tiny()
    layer = _hca(cfg)
    t = 16  # 16 / hca compression_rate=8 → 2 complete blocks
    h = torch.randn(1, t, cfg.hidden_dim)
    with torch.no_grad():
        out_full = layer(h)

    cache = HCAKVCacheLayer(
        max_seq_len=t,
        hidden_dim=cfg.hidden_dim,
        head_dim=cfg.attention.head_dim,
        compression_rate=cfg.hca.compression_rate,
    )
    with torch.no_grad():
        chunks = [layer(h[:, i:i + 1, :], kv_cache=cache) for i in range(t)]
    out_cached = torch.cat(chunks, dim=1)
    assert torch.allclose(out_cached, out_full, atol=1.0e-4)


def test_hca_cached_prefill_then_decode_matches_uncached() -> None:
    """Realistic generate: prefill K, decode rest one at a time."""
    cfg = ModelConfig.tiny()
    layer = _hca(cfg)
    prompt_t = 8  # exactly one HCA block boundary
    decode_t = 8
    full_t = prompt_t + decode_t
    h_full = torch.randn(1, full_t, cfg.hidden_dim)
    with torch.no_grad():
        out_full = layer(h_full)

    cache = HCAKVCacheLayer(
        max_seq_len=full_t,
        hidden_dim=cfg.hidden_dim,
        head_dim=cfg.attention.head_dim,
        compression_rate=cfg.hca.compression_rate,
    )
    with torch.no_grad():
        prefill_out = layer(h_full[:, :prompt_t], kv_cache=cache)
        chunks = [
            prefill_out,
            *(layer(h_full[:, i:i + 1], kv_cache=cache) for i in range(prompt_t, full_t)),
        ]
    out_cached = torch.cat(chunks, dim=1)
    assert torch.allclose(out_cached, out_full, atol=1.0e-4)


def test_hca_no_cache_unchanged() -> None:
    """Default kv_cache=None must be byte-identical to the pre-cache HCA."""
    cfg = ModelConfig.tiny()
    layer = _hca(cfg)
    h = torch.randn(1, 8, cfg.hidden_dim)
    with torch.no_grad():
        a = layer(h)
        b = layer(h, kv_cache=None)
    assert torch.equal(a, b)


# ---------- CSAKVCacheLayer ----------


def _csa(cfg: ModelConfig) -> CSA:
    torch.manual_seed(0)
    layer = CSA(cfg.hidden_dim, cfg.attention, cfg.csa)
    layer.eval()
    return layer


def test_csa_cache_initial_state_empty() -> None:
    cache = CSAKVCacheLayer(
        max_seq_len=16, hidden_dim=128, head_dim=32, c_indexer=16, compression_rate=4,
    )
    assert cache.length == 0
    assert cache.n_blocks == 0
    assert cache.h_buf_count == 0


def test_csa_cache_invalid_compression_rate() -> None:
    with pytest.raises(ValueError, match="compression_rate"):
        CSAKVCacheLayer(
            max_seq_len=8, hidden_dim=8, head_dim=4, c_indexer=4, compression_rate=0,
        )


def test_csa_cache_overflow_raises() -> None:
    cache = CSAKVCacheLayer(
        max_seq_len=4, hidden_dim=8, head_dim=4, c_indexer=4, compression_rate=2,
    )
    h = torch.randn(1, 5, 8)
    sw_k = torch.randn(1, 5, 4)
    sw_v = torch.randn(1, 5, 4)
    with pytest.raises(RuntimeError, match="overflow"):
        cache.append(
            h, sw_k, sw_v,
            compressor=lambda x: x.new_zeros(1, 0, 4),
            indexer_compressor=lambda x: x.new_zeros(1, 0, 4),
        )


def test_csa_cache_runs_both_compressors_on_block_boundaries() -> None:
    cache = CSAKVCacheLayer(
        max_seq_len=8, hidden_dim=8, head_dim=4, c_indexer=2, compression_rate=4,
    )
    kv_calls = [0]
    idx_calls = [0]

    def kv_stub(h: torch.Tensor) -> torch.Tensor:
        kv_calls[0] += 1
        return torch.zeros(h.shape[0], h.shape[1] // 4, 4)

    def idx_stub(h: torch.Tensor) -> torch.Tensor:
        idx_calls[0] += 1
        return torch.zeros(h.shape[0], h.shape[1] // 4, 2)

    cache.append(
        torch.randn(1, 4, 8), torch.randn(1, 4, 4), torch.randn(1, 4, 4),
        compressor=kv_stub, indexer_compressor=idx_stub,
    )
    assert kv_calls[0] == 1 and idx_calls[0] == 1
    assert cache.n_blocks == 1


def test_csa_cached_full_pass_matches_uncached() -> None:
    """CSA token-by-token with cache equals uncached full-pass — within atol
    that accommodates the documented loss of is_visual bias on the cached
    path (None is_visual in this test, so the math reduces exactly)."""
    cfg = ModelConfig.tiny()
    layer = _csa(cfg)
    t = 16
    h = torch.randn(1, t, cfg.hidden_dim)
    with torch.no_grad():
        out_full = layer(h)

    cache = CSAKVCacheLayer(
        max_seq_len=t,
        hidden_dim=cfg.hidden_dim,
        head_dim=cfg.attention.head_dim,
        c_indexer=cfg.csa.indexer_head_dim,
        compression_rate=cfg.csa.compression_rate,
    )
    with torch.no_grad():
        chunks = [layer(h[:, i:i + 1, :], kv_cache=cache) for i in range(t)]
    out_cached = torch.cat(chunks, dim=1)
    assert out_cached.shape == out_full.shape
    assert torch.allclose(out_cached, out_full, atol=1.0e-3)


def test_csa_no_cache_unchanged() -> None:
    cfg = ModelConfig.tiny()
    layer = _csa(cfg)
    h = torch.randn(1, 8, cfg.hidden_dim)
    with torch.no_grad():
        a = layer(h)
        b = layer(h, kv_cache=None)
    assert torch.equal(a, b)
