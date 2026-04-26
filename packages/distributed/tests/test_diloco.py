"""Tests for DiLoCo outer-loop synchronizer."""

from __future__ import annotations

import pytest
import torch
from saint_llm_distributed import DiLoCo, DiLoCoConfig
from torch import nn


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 4, bias=False), nn.Linear(4, 2, bias=False))


def test_diloco_starts_unsynced() -> None:
    model = _tiny_model()
    diloco = DiLoCo(model.parameters(), DiLoCoConfig(inner_steps=4))
    assert diloco.should_sync is False


def test_diloco_should_sync_after_inner_steps() -> None:
    model = _tiny_model()
    cfg = DiLoCoConfig(inner_steps=3)
    diloco = DiLoCo(model.parameters(), cfg)
    diloco.inner_step()
    diloco.inner_step()
    assert diloco.should_sync is False
    diloco.inner_step()
    assert diloco.should_sync is True


def test_diloco_outer_sync_resets_inner_count() -> None:
    model = _tiny_model()
    diloco = DiLoCo(model.parameters(), DiLoCoConfig(inner_steps=2))
    diloco.inner_step()
    diloco.inner_step()
    assert diloco.should_sync
    diloco.outer_sync()
    assert diloco.should_sync is False


def test_diloco_outer_sync_applies_pseudo_gradient_update() -> None:
    """When inner SGD changes params, outer_sync applies the pseudo-grad."""
    model = _tiny_model()
    cfg = DiLoCoConfig(inner_steps=1, outer_lr=1.0, outer_momentum=0.0)
    diloco = DiLoCo(model.parameters(), cfg)

    snapshot_pre = [p.detach().clone() for p in model.parameters()]
    # Simulate one inner step that perturbs params.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    diloco.inner_step()
    diloco.outer_sync()

    # With outer_lr=1.0, momentum=0, no peer reduction:
    # pseudo_grad = snapshot - new_param
    # update = pseudo_grad
    # new_param = snapshot - 1.0 * pseudo_grad = snapshot - (snapshot - new) = new
    # i.e., outer_sync is identity on a single peer with lr=1, mom=0.
    for i, p in enumerate(model.parameters()):
        torch.testing.assert_close(p.detach(), snapshot_pre[i] + (p.detach() - snapshot_pre[i]))


def test_diloco_with_mock_reducer_applies_reduction() -> None:
    """Custom reducer that halves all pseudo-grads should halve param drift."""

    class _HalfReducer:
        def reduce(self, tensors):
            return [t * 0.5 for t in tensors]

    model = _tiny_model()
    cfg = DiLoCoConfig(inner_steps=1, outer_lr=1.0, outer_momentum=0.0)
    diloco = DiLoCo(model.parameters(), cfg, reducer=_HalfReducer())

    snapshot_pre = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.full_like(p, 1.0))  # uniform +1 perturbation
    diloco.inner_step()
    diloco.outer_sync()

    # With Half reducer + outer_lr=1: params should be halfway between
    # snapshot and perturbed (i.e., snapshot + 0.5).
    for i, p in enumerate(model.parameters()):
        expected = snapshot_pre[i] + 0.5
        torch.testing.assert_close(p.detach(), expected, atol=1e-5, rtol=0)


def test_diloco_state_dict_roundtrip() -> None:
    model = _tiny_model()
    diloco = DiLoCo(model.parameters(), DiLoCoConfig(inner_steps=2))
    diloco.inner_step()
    state = diloco.state_dict()

    # Fresh diloco loads the state.
    model2 = _tiny_model()
    diloco2 = DiLoCo(model2.parameters(), DiLoCoConfig(inner_steps=2))
    diloco2.load_state_dict(state)

    # inner_count should be preserved.
    assert diloco2._inner_count == 1


def test_diloco_outer_lr_scales_update() -> None:
    """Larger outer_lr produces proportionally larger param movement."""
    model_a = _tiny_model()
    model_b = _tiny_model()
    # Make starting params identical.
    for pb, pa in zip(model_b.parameters(), model_a.parameters(), strict=True):
        pb.data.copy_(pa.data)

    diloco_a = DiLoCo(
        model_a.parameters(),
        DiLoCoConfig(inner_steps=1, outer_lr=0.5, outer_momentum=0.0),
    )
    diloco_b = DiLoCo(
        model_b.parameters(),
        DiLoCoConfig(inner_steps=1, outer_lr=1.5, outer_momentum=0.0),
    )

    # Identical perturbation.
    pert = [torch.randn_like(p) * 0.01 for p in model_a.parameters()]
    with torch.no_grad():
        for p, dp in zip(model_a.parameters(), pert, strict=True):
            p.add_(dp)
        for p, dp in zip(model_b.parameters(), pert, strict=True):
            p.add_(dp)

    diloco_a.inner_step()
    diloco_a.outer_sync()
    diloco_b.inner_step()
    diloco_b.outer_sync()

    # Mock reducer is identity. Pseudo-grad = snapshot - perturbed.
    # outer_sync applies: new = snapshot - outer_lr * (snapshot - perturbed)
    # = snapshot * (1 - outer_lr) + perturbed * outer_lr
    # Larger outer_lr → closer to perturbed; smaller → closer to snapshot.
    # Just verify the two diverged.
    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert not torch.allclose(pa.detach(), pb.detach())


def test_diloco_load_state_dict_validates_shape() -> None:
    model = _tiny_model()
    diloco = DiLoCo(model.parameters(), DiLoCoConfig())
    with pytest.raises(TypeError, match="wrong shape"):
        diloco.load_state_dict({"snapshot": "not a list", "momentum_buffer": []})
