"""Multimodal hooks + end-to-end SaintLLM forward smoke test on tiny config."""

from __future__ import annotations

import torch

from saint_llm_core import ModelConfig, SaintLLM
from saint_llm_core.multimodal import GenerationHeadHook, ResidualSideChannel


def test_side_channel_identity_at_zero_alpha() -> None:
    side = ResidualSideChannel(hidden_dim=16, alpha_init=0.0)
    h = torch.randn(2, 5, 16)
    out = side(h)
    assert torch.equal(out, h)


def test_generation_head_zero_init_yields_zero_logits() -> None:
    head = GenerationHeadHook(hidden_dim=16, vocab_slots=32, enabled=False)
    h = torch.randn(2, 5, 16)
    out = head(h)
    assert torch.all(out == 0.0)


def test_generation_head_frozen_when_disabled() -> None:
    head = GenerationHeadHook(hidden_dim=16, vocab_slots=32, enabled=False)
    for p in head.parameters():
        assert not p.requires_grad


def test_saintllm_forward_shapes() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    model.eval()
    t = 16  # divisible by csa.m=2 and hca.m'=8
    token_ids = torch.randint(0, cfg.vocab_size, (2, t))
    with torch.no_grad():
        out = model(token_ids)
    assert out["logits"].shape == (2, t, cfg.vocab_size)
    assert out["hidden"].shape == (2, t, cfg.hidden_dim)
    assert isinstance(out["mtp_logits"], list)
    assert len(out["mtp_logits"]) == cfg.mtp.depth
    assert out["mtp_logits"][0].shape == (2, t, cfg.vocab_size)


def test_saintllm_forward_finite() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    model.eval()
    t = 16
    token_ids = torch.randint(0, cfg.vocab_size, (2, t))
    with torch.no_grad():
        out = model(token_ids)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["hidden"]).all()


def test_saintllm_backward_pass() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    t = 16
    token_ids = torch.randint(0, cfg.vocab_size, (2, t))
    out = model(token_ids)
    loss = out["logits"].mean() + out["mtp_logits"][0].mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    assert has_grad


def test_saintllm_generation_head_zero_when_disabled() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    model.eval()
    t = 16
    token_ids = torch.randint(0, cfg.vocab_size, (2, t))
    with torch.no_grad():
        out = model(token_ids)
    assert torch.all(out["generation_logits"] == 0.0)
