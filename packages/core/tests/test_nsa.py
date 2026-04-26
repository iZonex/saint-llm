"""Tests for NSA — Native Sparse Attention (ARCH-03 / arXiv 2502.11089)."""

from __future__ import annotations

import torch
from saint_llm_core.attention import NSAttention
from saint_llm_core.attention.nsa import _block_compress
from saint_llm_core.config import AttentionConfig, ModelConfig


def _tiny_attn() -> AttentionConfig:
    return AttentionConfig(
        query_heads=4,
        head_dim=16,
        query_compression_dim=32,
        output_proj_groups=2,
        attention_intermediate_dim=32,
        sliding_window_size=4,
        rope_dim=8,
        use_attention_sink=False,
    )


def test_block_compress_drops_partial_block() -> None:
    x = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
    out = _block_compress(x, block_size=3)
    # 7 // 3 = 2 full blocks; trailing token dropped.
    assert out.shape == (2, 2, 3)
    # Block 0 = mean of x[:, 0:3] along dim -2.
    assert torch.allclose(out[:, 0], x[:, 0:3].mean(-2))
    assert torch.allclose(out[:, 1], x[:, 3:6].mean(-2))


def test_block_compress_zero_blocks_when_t_below_block_size() -> None:
    x = torch.randn(1, 2, 4, 5)
    out = _block_compress(x, block_size=8)
    assert out.shape == (1, 2, 0, 5)


def test_nsa_forward_shape() -> None:
    attn = _tiny_attn()
    nsa = NSAttention(
        hidden_dim=64, attn=attn, compression_rate=4, selection_k=2,
    )
    h = torch.randn(2, 16, 64)
    out = nsa(h)
    assert out.shape == h.shape


def test_nsa_forward_short_seq_falls_back_to_window_only() -> None:
    """When T < compression_rate, compressed/selective branches are zero."""
    attn = _tiny_attn()
    nsa = NSAttention(
        hidden_dim=64, attn=attn, compression_rate=8, selection_k=2,
    )
    h = torch.randn(1, 4, 64)  # T=4 < cr=8
    out = nsa(h)
    assert out.shape == h.shape
    assert torch.isfinite(out).all()


def test_nsa_qk_clip_targets_returns_q_and_k_weights() -> None:
    nsa = NSAttention(
        hidden_dim=64, attn=_tiny_attn(), compression_rate=4, selection_k=2,
    )
    targets = nsa.qk_clip_targets()
    assert len(targets) == 2
    assert targets[0] is nsa.q_up.weight
    assert targets[1] is nsa.k_proj.weight


def test_nsa_max_attn_logit_tracked() -> None:
    nsa = NSAttention(
        hidden_dim=64, attn=_tiny_attn(), compression_rate=4, selection_k=2,
    )
    h = torch.randn(1, 16, 64)
    nsa(h)
    # MuonClip telemetry must be populated after a forward.
    assert nsa._last_max_attn_logit > 0.0


def test_nsa_gate_starts_equal_mix() -> None:
    """Zero-init weight + bias -> sigmoid(0)=0.5 then normalize -> 1/3 each."""
    nsa = NSAttention(
        hidden_dim=64, attn=_tiny_attn(), compression_rate=4, selection_k=2,
    )
    q_dummy = torch.zeros(1, 4, 1, 16)  # (B, H, T, head_dim)
    gates = torch.sigmoid(nsa.branch_gate(q_dummy))
    gates = gates / gates.sum(dim=-1, keepdim=True)
    assert torch.allclose(gates, torch.full_like(gates, 1.0 / 3.0))


def test_nsa_causality_compressed_branch_only_sees_past() -> None:
    """A future-position perturbation must not change the output at earlier positions."""
    torch.manual_seed(0)
    attn = _tiny_attn()
    nsa = NSAttention(
        hidden_dim=64, attn=attn, compression_rate=4, selection_k=2,
    )
    nsa.eval()
    h = torch.randn(1, 16, 64)
    h2 = h.clone()
    h2[0, 12:] = torch.randn(4, 64)  # change tokens 12..15
    with torch.no_grad():
        out1 = nsa(h)
        out2 = nsa(h2)
    # Output at positions 0..7 should be identical: window=4 so positions 0..7
    # never see token 12+. Compressed branch needs blocks ending <= q_pos so
    # at q_pos=7 it only sees blocks ending at 3 and 7 (start tokens 0..7).
    assert torch.allclose(out1[0, :8], out2[0, :8], atol=1e-5)


def test_nsa_with_sink_logit_works() -> None:
    attn = AttentionConfig(
        query_heads=4,
        head_dim=16,
        query_compression_dim=32,
        output_proj_groups=2,
        attention_intermediate_dim=32,
        sliding_window_size=4,
        rope_dim=8,
        use_attention_sink=True,
    )
    nsa = NSAttention(hidden_dim=64, attn=attn, compression_rate=4, selection_k=2)
    assert nsa.sink_logit is not None
    h = torch.randn(1, 16, 64)
    out = nsa(h)
    assert out.shape == h.shape


def test_nsa_with_logit_softcap_runs() -> None:
    attn = AttentionConfig(
        query_heads=4,
        head_dim=16,
        query_compression_dim=32,
        output_proj_groups=2,
        attention_intermediate_dim=32,
        sliding_window_size=4,
        rope_dim=8,
        use_attention_sink=False,
        logit_softcap=50.0,
    )
    nsa = NSAttention(hidden_dim=64, attn=attn, compression_rate=4, selection_k=2)
    h = torch.randn(1, 16, 64)
    out = nsa(h)
    assert torch.isfinite(out).all()


def test_nsa_grad_flows_through_all_three_branches() -> None:
    attn = _tiny_attn()
    nsa = NSAttention(
        hidden_dim=64, attn=attn, compression_rate=4, selection_k=2,
    )
    h = torch.randn(1, 16, 64, requires_grad=True)
    out = nsa(h)
    out.sum().backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    # Gate weights should also receive gradient (trained jointly).
    assert nsa.branch_gate.weight.grad is not None
    assert torch.isfinite(nsa.branch_gate.weight.grad).all()


def test_nsa_selective_k_zero_falls_back() -> None:
    """selection_k=0 -> selective branch contributes zero; no error."""
    nsa = NSAttention(
        hidden_dim=64, attn=_tiny_attn(), compression_rate=4, selection_k=0,
    )
    h = torch.randn(1, 16, 64)
    out = nsa(h)
    assert out.shape == h.shape


def test_nsa_with_modelconfig_tiny_attn_dims() -> None:
    """Smoke test with the project's standard tiny attention dims."""
    cfg = ModelConfig.tiny()
    nsa = NSAttention(
        hidden_dim=cfg.hidden_dim,
        attn=cfg.attention,
        compression_rate=4,
        selection_k=2,
    )
    h = torch.randn(2, 16, cfg.hidden_dim)
    out = nsa(h)
    assert out.shape == h.shape
