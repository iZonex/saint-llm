"""Shape + causality tests for SWAttention, CSA, HCA."""

from __future__ import annotations

import torch

from saint_llm_core.attention import CSA, HCA, SWAttention
from saint_llm_core.attention.common import causal_mask, sliding_window_mask
from saint_llm_core.config import ModelConfig


def test_causal_mask_shape_and_values() -> None:
    m = causal_mask(4, 4)
    expected = torch.tensor(
        [[True, False, False, False],
         [True, True, False, False],
         [True, True, True, False],
         [True, True, True, True]],
    )
    assert torch.equal(m, expected)


def test_sliding_window_mask() -> None:
    m = sliding_window_mask(5, window=2)
    # Position 4 should see positions 3 and 4 only (window=2 includes self).
    assert m[4, 4] and m[4, 3] and not m[4, 2]
    # Position 0 only sees itself.
    assert m[0, 0] and not m[0, 1]


def _seq_len_aligned_to_compression(cfg: ModelConfig) -> int:
    """Seq length divisible by both CSA.m and HCA.m'."""
    import math
    return math.lcm(cfg.csa.compression_rate, cfg.hca.compression_rate)


def test_swa_forward_shape() -> None:
    cfg = ModelConfig.tiny()
    swa = SWAttention(cfg.hidden_dim, cfg.attention)
    t = 16
    h = torch.randn(2, t, cfg.hidden_dim)
    out = swa(h)
    assert out.shape == h.shape


def test_csa_forward_shape() -> None:
    cfg = ModelConfig.tiny()
    csa = CSA(cfg.hidden_dim, cfg.attention, cfg.csa)
    t = _seq_len_aligned_to_compression(cfg)
    h = torch.randn(2, t, cfg.hidden_dim)
    out = csa(h)
    assert out.shape == h.shape


def test_hca_forward_shape() -> None:
    cfg = ModelConfig.tiny()
    hca = HCA(cfg.hidden_dim, cfg.attention, cfg.hca)
    t = _seq_len_aligned_to_compression(cfg)
    h = torch.randn(2, t, cfg.hidden_dim)
    out = hca(h)
    assert out.shape == h.shape


def test_csa_with_visual_mask_runs() -> None:
    cfg = ModelConfig.tiny()
    csa = CSA(cfg.hidden_dim, cfg.attention, cfg.csa)
    t = _seq_len_aligned_to_compression(cfg)
    h = torch.randn(2, t, cfg.hidden_dim)
    is_visual = torch.zeros(2, t, dtype=torch.bool)
    is_visual[0, 4:8] = True
    out = csa(h, is_visual=is_visual)
    assert out.shape == h.shape


def test_swa_strict_causality() -> None:
    """A change at position t must not affect output at positions < t."""
    cfg = ModelConfig.tiny()
    swa = SWAttention(cfg.hidden_dim, cfg.attention)
    swa.eval()
    t = 12
    h1 = torch.randn(1, t, cfg.hidden_dim)
    h2 = h1.clone()
    h2[0, 8] += 5.0
    with torch.no_grad():
        o1 = swa(h1)
        o2 = swa(h2)
    # Positions before 8 should be identical.
    diff = (o1[0, :8] - o2[0, :8]).abs().max().item()
    assert diff < 1.0e-4, f"Causality violated: max diff at positions <8 = {diff}"


def test_csa_strict_causality() -> None:
    """Regression: indexer used to pick causally-future blocks via -inf-tail topk
    when valid_count < top_k. Mutating a future position must not leak back."""
    cfg = ModelConfig.tiny()
    csa = CSA(cfg.hidden_dim, cfg.attention, cfg.csa)
    csa.eval()
    t = _seq_len_aligned_to_compression(cfg)
    h1 = torch.randn(1, t, cfg.hidden_dim)
    mutate_at = t - 2
    h2 = h1.clone()
    h2[0, mutate_at] += 5.0
    with torch.no_grad():
        o1 = csa(h1)
        o2 = csa(h2)
    diff = (o1[0, :mutate_at] - o2[0, :mutate_at]).abs().max().item()
    assert diff < 1.0e-4, f"CSA causality violated: max diff at positions <{mutate_at} = {diff}"
    assert torch.isfinite(o1).all()


def test_csa_no_nan_at_zero_position() -> None:
    """Position 0 has no completed compressed blocks → all indexer picks are -1.
    Sparse attention must safely produce 0 contribution (not NaN)."""
    cfg = ModelConfig.tiny()
    csa = CSA(cfg.hidden_dim, cfg.attention, cfg.csa)
    csa.eval()
    t = _seq_len_aligned_to_compression(cfg)
    h = torch.randn(2, t, cfg.hidden_dim)
    with torch.no_grad():
        out = csa(h)
    assert torch.isfinite(out).all()
