"""Tests for MuonClip QK-Clip post-step rescale (ADR-0010 / OPT-01)."""

from __future__ import annotations

import pytest
import torch
from saint_llm_optim.muon import Muon
from torch import Tensor, nn


class _FakeAttention(nn.Module):
    """Minimal stand-in matching the QK-clip protocol for unit tests.

    Has Q (q_up) and K (k_proj) weight matrices that the optimizer can
    rescale. Stores ``_last_max_attn_logit`` that the test sets directly
    to simulate a forward pass with a desired logit magnitude.
    """

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.q_up = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        # Add at least one Muon-eligible param to a different module so the
        # optimizer has something to step on (the q_up/k_proj weights live
        # under this module too — Muon will pick them up via params arg).
        self._last_max_attn_logit: float = 0.0

    def qk_clip_targets(self) -> list[Tensor]:
        return [self.q_up.weight, self.k_proj.weight]


def _muon_with_layer(layer: _FakeAttention, **kwargs: object) -> Muon:
    # Pass q_up + k_proj as params so optimizer can step them; not strictly
    # necessary for the QK-clip test but keeps Muon happy with non-empty
    # param groups.
    return Muon(
        [layer.q_up.weight, layer.k_proj.weight],
        lr=1e-3,
        qk_clip_layers=[layer],
        **kwargs,
    )


def test_register_qk_clip_layer_rejects_non_clippable() -> None:
    bad = nn.Linear(8, 8)  # no _last_max_attn_logit, no qk_clip_targets
    opt = Muon([bad.weight], lr=1e-3, qk_clip_layers=())
    with pytest.raises(TypeError, match="_QKClippable"):
        opt.register_qk_clip_layer(bad)


def test_below_tau_no_rescale() -> None:
    layer = _FakeAttention(hidden=8)
    opt = _muon_with_layer(layer, qk_clip_tau=100.0)

    layer._last_max_attn_logit = 50.0  # below tau
    q_pre = layer.q_up.weight.detach().clone()
    k_pre = layer.k_proj.weight.detach().clone()

    # Run only the QK-clip pass (not a full optimizer step that would
    # require gradients).
    opt._apply_qk_clip()

    torch.testing.assert_close(layer.q_up.weight, q_pre)
    torch.testing.assert_close(layer.k_proj.weight, k_pre)
    assert opt.qk_clip_count == {}


def test_above_tau_rescales_q_and_k() -> None:
    layer = _FakeAttention(hidden=8)
    opt = _muon_with_layer(layer, qk_clip_tau=100.0)

    layer._last_max_attn_logit = 1000.0  # 10x above tau
    q_pre = layer.q_up.weight.detach().clone()
    k_pre = layer.k_proj.weight.detach().clone()
    expected_scale = (100.0 / 1000.0) ** 0.5  # ~0.316

    opt._apply_qk_clip()

    torch.testing.assert_close(layer.q_up.weight, q_pre * expected_scale)
    torch.testing.assert_close(layer.k_proj.weight, k_pre * expected_scale)
    assert opt.qk_clip_count == {id(layer): 1}


def test_clip_count_increments_across_calls() -> None:
    layer = _FakeAttention(hidden=8)
    opt = _muon_with_layer(layer, qk_clip_tau=100.0)
    layer._last_max_attn_logit = 500.0
    opt._apply_qk_clip()
    opt._apply_qk_clip()
    opt._apply_qk_clip()
    assert opt.qk_clip_count[id(layer)] == 3


def test_disabled_skips_apply_in_step() -> None:
    """When qk_clip_enabled=False, step() does not invoke apply."""
    layer = _FakeAttention(hidden=8)
    opt = Muon(
        [layer.q_up.weight, layer.k_proj.weight],
        lr=1e-3,
        qk_clip_enabled=False,
        qk_clip_layers=[layer],
        qk_clip_tau=100.0,
    )
    layer._last_max_attn_logit = 1000.0  # above tau
    q_pre = layer.q_up.weight.detach().clone()

    # Need grads for step() to do anything; provide synthetic.
    layer.q_up.weight.grad = torch.zeros_like(layer.q_up.weight)
    layer.k_proj.weight.grad = torch.zeros_like(layer.k_proj.weight)
    opt.step()

    # With disabled flag, no rescale; weights only updated by Muon
    # gradient step (which is zero here so weights ~unchanged modulo
    # weight decay).
    decay_factor = 1.0 - 1e-3 * 0.1  # default wd=0.1
    torch.testing.assert_close(layer.q_up.weight, q_pre * decay_factor)
    assert opt.qk_clip_count == {}


def test_step_applies_qk_clip_after_param_update() -> None:
    """Full step() with grad: Muon update happens first, then QK-clip rescale."""
    layer = _FakeAttention(hidden=8)
    opt = Muon(
        [layer.q_up.weight, layer.k_proj.weight],
        lr=1e-3,
        qk_clip_enabled=True,
        qk_clip_layers=[layer],
        qk_clip_tau=100.0,
    )
    layer._last_max_attn_logit = 400.0  # 4x above tau

    layer.q_up.weight.grad = torch.zeros_like(layer.q_up.weight)
    layer.k_proj.weight.grad = torch.zeros_like(layer.k_proj.weight)

    q_pre = layer.q_up.weight.detach().clone()
    opt.step()

    # post-Muon: q_pre * (1 - lr*wd)
    decay_factor = 1.0 - 1e-3 * 0.1
    expected_after_muon = q_pre * decay_factor
    # post-QK-clip: * sqrt(tau / max_logit) = sqrt(100/400) = 0.5
    expected_final = expected_after_muon * 0.5

    torch.testing.assert_close(layer.q_up.weight, expected_final)
    assert opt.qk_clip_count[id(layer)] == 1


def test_invalid_tau_raises() -> None:
    layer = _FakeAttention(hidden=8)
    with pytest.raises(ValueError, match="qk_clip_tau must be positive"):
        Muon(
            [layer.q_up.weight],
            lr=1e-3,
            qk_clip_tau=0.0,
            qk_clip_layers=[layer],
        )


def test_register_qk_clip_layer_appends() -> None:
    layer1 = _FakeAttention(hidden=8)
    layer2 = _FakeAttention(hidden=8)
    opt = Muon([layer1.q_up.weight], lr=1e-3, qk_clip_layers=[layer1])
    opt.register_qk_clip_layer(layer2)

    layer1._last_max_attn_logit = 200.0
    layer2._last_max_attn_logit = 200.0
    opt._apply_qk_clip()

    assert opt.qk_clip_count[id(layer1)] == 1
    assert opt.qk_clip_count[id(layer2)] == 1
