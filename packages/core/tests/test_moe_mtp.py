"""MoE + MTP tests: routing balance, hash determinism, modality bias zero-init, MTP shapes."""

from __future__ import annotations

import torch

from saint_llm_core.config import ModelConfig
from saint_llm_core.moe import DeepSeekMoE, HashRouter
from saint_llm_core.mtp import MTPStack


def test_hash_router_deterministic() -> None:
    token_ids = torch.tensor([[5, 12, 100, 7]])
    idx_a, w_a = HashRouter.route(token_ids, n_experts=8, k=2)
    idx_b, w_b = HashRouter.route(token_ids, n_experts=8, k=2)
    assert torch.equal(idx_a, idx_b)
    assert torch.equal(w_a, w_b)


def test_hash_router_no_collisions_within_token() -> None:
    """Top-k must select distinct experts per token."""
    token_ids = torch.arange(50).unsqueeze(0)
    idx, _w = HashRouter.route(token_ids, n_experts=8, k=4)
    for t in range(50):
        assert len(set(idx[0, t].tolist())) == 4, f"Token {t} got duplicate experts: {idx[0, t]}"


def test_moe_hash_routed_layer_runs() -> None:
    cfg = ModelConfig.tiny()
    moe = DeepSeekMoE(cfg.hidden_dim, cfg.moe, layer_idx=0)  # hash-routed
    h = torch.randn(2, 8, cfg.hidden_dim)
    token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = moe(h, token_ids=token_ids)
    assert out.shape == h.shape
    assert moe.use_hash_routing is True


def test_moe_learned_routed_layer_runs() -> None:
    cfg = ModelConfig.tiny()
    moe = DeepSeekMoE(cfg.hidden_dim, cfg.moe, layer_idx=cfg.moe.hash_routed_layers + 1)
    h = torch.randn(2, 8, cfg.hidden_dim)
    token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = moe(h, token_ids=token_ids)
    assert out.shape == h.shape
    assert moe.use_hash_routing is False


def test_moe_modality_bias_zero_init() -> None:
    cfg = ModelConfig.tiny()
    moe = DeepSeekMoE(cfg.hidden_dim, cfg.moe, layer_idx=cfg.moe.hash_routed_layers + 1)
    assert moe.router is not None
    assert moe.router.modality_bias is not None
    assert torch.all(moe.router.modality_bias == 0.0)


def test_moe_with_visual_routing_runs() -> None:
    cfg = ModelConfig.tiny()
    moe = DeepSeekMoE(cfg.hidden_dim, cfg.moe, layer_idx=cfg.moe.hash_routed_layers + 1)
    h = torch.randn(2, 8, cfg.hidden_dim)
    token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    is_visual = torch.tensor([[True, False, True, False, False, True, False, False],
                              [False, False, False, False, False, False, False, False]])
    out = moe(h, token_ids=token_ids, is_visual=is_visual)
    assert out.shape == h.shape


def test_mtp_depth_one_default() -> None:
    cfg = ModelConfig.tiny()
    assert cfg.mtp.depth == 1
    mtp = MTPStack(cfg.hidden_dim, cfg.vocab_size, cfg.mtp)
    h = torch.randn(2, 6, cfg.hidden_dim)
    embeds = torch.randn(2, 6, cfg.hidden_dim)
    out = mtp(h, embeds)
    assert len(out) == 1
    assert out[0].shape == h.shape
