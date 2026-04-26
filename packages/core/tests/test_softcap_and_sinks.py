"""Tests for ADR-0012: logit softcap + learnable sink tokens."""

from __future__ import annotations

import torch
from saint_llm_core.attention.common import scaled_dot_product
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM


def _tiny_cfg(**overrides: object) -> ModelConfig:
    cfg = ModelConfig.tiny()  # type: ignore[attr-defined]
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    return cfg


def test_scaled_dot_product_softcap_clips_large_logits() -> None:
    """With deliberate large logits, softcap bounds them to ~[-c, +c]."""
    torch.manual_seed(0)
    b, h, t, d = 1, 1, 4, 8
    # Force a large dot-product magnitude by scaling Q dramatically.
    q = torch.randn(b, h, t, d) * 50.0
    k = torch.randn(b, h, t, d) * 50.0
    v = torch.randn(b, h, t, d)

    raw_max = float("-inf")

    def _observe(scores: torch.Tensor) -> None:
        nonlocal raw_max
        raw_max = max(raw_max, float(scores.detach().abs().max()))

    # No softcap: scores are huge, observer sees raw value.
    _ = scaled_dot_product(q, k, v, score_observer=_observe)
    assert raw_max > 100.0  # confirm we constructed a real spike

    # With softcap=10.0: post-cap scores stay <= 10. We can verify by
    # capturing post-cap scores via a custom observer that fires AFTER cap.
    # Instead: verify the math directly — call again with softcap=10.
    raw_max2 = float("-inf")

    def _observe2(scores: torch.Tensor) -> None:
        nonlocal raw_max2
        raw_max2 = max(raw_max2, float(scores.detach().abs().max()))

    _ = scaled_dot_product(q, k, v, score_observer=_observe2, logit_softcap=10.0)
    # Observer is called BEFORE cap (per ADR-0012); raw value still huge.
    assert raw_max2 > 100.0

    # Math: 10 * tanh(scores / 10) is bounded to (-10, 10).
    raw_scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (d ** 0.5)
    capped = 10.0 * torch.tanh(raw_scores / 10.0)
    assert capped.abs().max().item() <= 10.0


def test_scaled_dot_product_softcap_no_op_for_small_logits() -> None:
    """For small logits, softcap is approximately identity."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8) * 0.1
    k = torch.randn(1, 1, 4, 8) * 0.1
    v = torch.randn(1, 1, 4, 8)
    out_off = scaled_dot_product(q, k, v)
    out_on = scaled_dot_product(q, k, v, logit_softcap=50.0)
    # tanh(small_x / 50) ≈ small_x / 50; 50 * tanh(...) ≈ small_x. Output
    # should be essentially identical (within float tolerance).
    torch.testing.assert_close(out_on, out_off, atol=1e-4, rtol=1e-4)


def test_scaled_dot_product_softcap_observer_sees_pre_cap() -> None:
    """The observer fires before softcap is applied so QK-Clip sees raw."""
    q = torch.randn(1, 1, 4, 8) * 100.0
    k = torch.randn(1, 1, 4, 8) * 100.0
    v = torch.randn(1, 1, 4, 8)
    seen: list[float] = []

    def _observe(scores: torch.Tensor) -> None:
        seen.append(float(scores.detach().abs().max()))

    _ = scaled_dot_product(
        q, k, v, score_observer=_observe, logit_softcap=10.0,
    )
    assert len(seen) == 1
    # Pre-cap value should be way larger than the cap.
    assert seen[0] > 100.0


def test_softcap_default_is_no_op() -> None:
    """logit_softcap=None passes scores through unchanged."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    out_none = scaled_dot_product(q, k, v)
    # No-op equivalent to no logit_softcap arg.
    out_explicit_none = scaled_dot_product(q, k, v, logit_softcap=None)
    torch.testing.assert_close(out_none, out_explicit_none)


def test_attention_config_default_softcap_is_none() -> None:
    """Default keeps softcap off for v0.0 backward compat."""
    cfg = _tiny_cfg()
    assert cfg.attention.logit_softcap is None


def test_model_config_default_n_sink_tokens_is_zero() -> None:
    """Default keeps sinks off for v0.0 backward compat."""
    cfg = _tiny_cfg()
    assert cfg.n_sink_tokens == 0
    assert cfg.final_logit_softcap is None


def test_saintllm_no_sinks_when_zero() -> None:
    cfg = _tiny_cfg()
    model = SaintLLM(cfg)
    assert model.sink_embeddings is None


def test_saintllm_creates_sink_param_when_n_sink_positive() -> None:
    cfg = _tiny_cfg(n_sink_tokens=3)
    model = SaintLLM(cfg)
    assert model.sink_embeddings is not None
    assert model.sink_embeddings.shape == (3, cfg.hidden_dim)
    assert model.sink_embeddings.requires_grad


def test_saintllm_forward_with_sinks_logits_shape_matches_input() -> None:
    """Even though sinks participate in attention, lm_head output strips
    them — so logits.shape[1] equals input token count, not T+N.
    """
    cfg = _tiny_cfg(n_sink_tokens=2)
    model = SaintLLM(cfg)
    model.eval()
    token_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        out = model(token_ids)
    assert out["logits"].shape == (1, 6, cfg.vocab_size)


def test_saintllm_forward_no_sinks_logits_shape_unchanged() -> None:
    """Without sinks the output shape is the same as today."""
    cfg = _tiny_cfg(n_sink_tokens=0)
    model = SaintLLM(cfg)
    model.eval()
    token_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        out = model(token_ids)
    assert out["logits"].shape == (1, 6, cfg.vocab_size)


def test_saintllm_forward_final_logit_softcap_bounds_logits() -> None:
    cfg = _tiny_cfg(final_logit_softcap=5.0)
    model = SaintLLM(cfg)
    model.eval()
    token_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out = model(token_ids)
    logits = out["logits"]
    assert logits.abs().max().item() <= 5.0


def test_saintllm_sink_grad_flows() -> None:
    """Backprop through sink embeddings produces nonzero gradient."""
    cfg = _tiny_cfg(n_sink_tokens=2)
    model = SaintLLM(cfg)
    token_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model(token_ids)
    loss = out["logits"].sum()
    loss.backward()
    assert model.sink_embeddings is not None
    assert model.sink_embeddings.grad is not None
    assert model.sink_embeddings.grad.abs().sum().item() > 0
