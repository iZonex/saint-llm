"""Tests for trustless gradient verification."""

from __future__ import annotations

import pytest
import torch
from saint_llm_distributed.verification import (
    GradientCheck,
    GradientVerifier,
    VerificationResult,
    random_peer_gradient,
)
from torch import nn


def _tiny_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))


def _mse_loss(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
) -> torch.Tensor:
    return ((model(x) - y) ** 2).mean()


def _compute_local_grad(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
) -> tuple[list[torch.Tensor], list[str]]:
    """Helper: compute gradients on x, y and return (grads, names) lists."""
    model.train()
    for p in model.parameters():
        p.grad = None
    loss = _mse_loss(model, x, y)
    loss.backward()
    grads = []
    names = []
    for name, p in model.named_parameters():
        grads.append(p.grad.detach().clone())  # type: ignore[union-attr]
        names.append(name)
    return grads, names


def test_honest_peer_accepts() -> None:
    """When peer submits the actual local gradient, verifier accepts."""
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)

    # Peer's "claim" is just the result of backward on the same batch
    # under the same model.
    peer_model = _tiny_model()
    peer_model.load_state_dict(model.state_dict())
    grads, names = _compute_local_grad(peer_model, x, y)

    verifier = GradientVerifier(model, _mse_loss, cosine_threshold=0.99)
    result = verifier.verify(grads, names, inputs=x, labels=y)

    assert result.accepted is True
    assert result.mean_cosine > 0.99
    assert "accepted" in result.reason


def test_random_peer_rejects() -> None:
    """Garbage gradient claim -> verifier rejects."""
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)

    grads, names = random_peer_gradient(model, scale=0.1, seed=42)
    verifier = GradientVerifier(model, _mse_loss, cosine_threshold=0.95)
    result = verifier.verify(grads, names, inputs=x, labels=y)

    assert result.accepted is False
    # Random gradient should have ~0 cosine similarity on average.
    assert abs(result.mean_cosine) < 0.5


def test_inverted_peer_rejects() -> None:
    """Negated honest gradient (cos = -1) is the worst case."""
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)

    peer_model = _tiny_model()
    peer_model.load_state_dict(model.state_dict())
    honest_grads, names = _compute_local_grad(peer_model, x, y)
    inverted = [-g for g in honest_grads]

    verifier = GradientVerifier(model, _mse_loss, cosine_threshold=0.95)
    result = verifier.verify(inverted, names, inputs=x, labels=y)

    assert result.accepted is False
    assert result.min_cosine < -0.99


def test_high_threshold_rejects_noisy_peer() -> None:
    """Slightly noisy honest gradient passes a lenient threshold but not strict."""
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)

    peer_model = _tiny_model()
    peer_model.load_state_dict(model.state_dict())
    honest_grads, names = _compute_local_grad(peer_model, x, y)
    # Add ~10% noise so cosine is high but not 1.0.
    noisy = [
        g + torch.randn_like(g) * 0.1 * g.abs().max()
        for g in honest_grads
    ]

    lenient = GradientVerifier(model, _mse_loss, cosine_threshold=0.5)
    strict = GradientVerifier(model, _mse_loss, cosine_threshold=0.999)
    assert lenient.verify(noisy, names, inputs=x, labels=y).accepted is True
    assert strict.verify(noisy, names, inputs=x, labels=y).accepted is False


def test_verifier_preserves_existing_grads() -> None:
    """Verification must not trample the verifier model's accumulated grads."""
    torch.manual_seed(0)
    model = _tiny_model()
    # Stamp a known gradient onto the model.
    sentinel = {
        name: torch.full_like(p, 7.0)
        for name, p in model.named_parameters()
    }
    for name, p in model.named_parameters():
        p.grad = sentinel[name].clone()

    x = torch.randn(2, 8)
    y = torch.randn(2, 2)
    grads, names = random_peer_gradient(model, scale=0.0, seed=0)
    GradientVerifier(model, _mse_loss).verify(grads, names, inputs=x, labels=y)

    # Original grads should be intact.
    for name, p in model.named_parameters():
        assert torch.allclose(p.grad, sentinel[name])  # type: ignore[arg-type]


def test_param_filter_skips_excluded_names() -> None:
    """Only filtered parameters contribute to the verification decision."""
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)

    # Build a peer claim where the first param is honest but the rest is garbage.
    peer_model = _tiny_model()
    peer_model.load_state_dict(model.state_dict())
    honest, names = _compute_local_grad(peer_model, x, y)
    mixed = [honest[0]] + [torch.randn_like(g) for g in honest[1:]]

    # Filter to only the first param -> verifier sees only honest piece.
    only_first = GradientVerifier(
        model, _mse_loss, cosine_threshold=0.99,
        param_filter=lambda n: n == names[0],
    )
    assert only_first.verify(mixed, names, inputs=x, labels=y).accepted is True


def test_verifier_no_matching_params_rejects() -> None:
    """If no peer params match the model, verification fails."""
    torch.manual_seed(0)
    model = _tiny_model()
    grads = [torch.randn(8, 8)]
    names = ["completely.unknown.param"]
    result = GradientVerifier(model, _mse_loss).verify(
        grads, names, inputs=torch.randn(2, 8), labels=torch.randn(2, 2),
    )
    assert result.accepted is False
    assert result.checks == ()


def test_verifier_shape_mismatch_raises() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    # Use a real model param name but wrong shape.
    name = next(iter(dict(model.named_parameters())))
    bad_grad = torch.randn(99, 99)
    with pytest.raises(ValueError, match="shape mismatch"):
        GradientVerifier(model, _mse_loss).verify(
            [bad_grad], [name],
            inputs=torch.randn(2, 8), labels=torch.randn(2, 2),
        )


def test_verifier_grads_names_length_mismatch_raises() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="length"):
        GradientVerifier(model, _mse_loss).verify(
            [torch.zeros(1)], ["a", "b"],
            inputs=torch.randn(2, 8), labels=torch.randn(2, 2),
        )


def test_verifier_threshold_validated() -> None:
    with pytest.raises(ValueError, match="cosine_threshold"):
        GradientVerifier(_tiny_model(), _mse_loss, cosine_threshold=1.5)


def test_verification_result_reason_strings() -> None:
    """Reason field describes why a result was accepted/rejected."""
    accepted = VerificationResult(
        accepted=True, mean_cosine=0.99, min_cosine=0.95,
        checks=(GradientCheck(name="x", cosine=0.99, norm_ratio=1.0),),
    )
    rejected = VerificationResult(
        accepted=False, mean_cosine=0.5, min_cosine=-0.1,
        checks=(GradientCheck(name="x", cosine=-0.1, norm_ratio=1.0),),
    )
    assert "accepted" in accepted.reason
    assert "rejected" in rejected.reason


def test_random_peer_gradient_reproducible_with_seed() -> None:
    model = _tiny_model()
    g1, n1 = random_peer_gradient(model, scale=1.0, seed=7)
    g2, n2 = random_peer_gradient(model, scale=1.0, seed=7)
    assert n1 == n2
    for a, b in zip(g1, g2, strict=True):
        assert torch.equal(a, b)


def test_random_peer_gradient_matches_param_shapes() -> None:
    model = _tiny_model()
    grads, names = random_peer_gradient(model, scale=1.0)
    name_to_shape = {n: p.shape for n, p in model.named_parameters()}
    for name, grad in zip(names, grads, strict=True):
        assert grad.shape == name_to_shape[name]


def test_disable_dropout_during_verification() -> None:
    """Verifier sets eval mode internally — dropout shouldn't affect cosine."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 4), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4, 2),
    )

    def loss(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ((m(x) - y) ** 2).mean()

    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    peer = nn.Sequential(*[
        nn.Linear(8, 4), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4, 2),
    ])
    peer.load_state_dict(model.state_dict())
    peer.eval()
    for p in peer.parameters():
        p.grad = None
    loss(peer, x, y).backward()
    peer_grads = [p.grad.detach().clone() for p in peer.parameters()]  # type: ignore[union-attr]
    names = [n for n, _ in peer.named_parameters()]

    result = GradientVerifier(
        model, loss, cosine_threshold=0.99,
    ).verify(peer_grads, names, inputs=x, labels=y)
    assert result.accepted is True


def test_norm_ratio_is_finite_for_zero_input() -> None:
    """Edge case: zero gradients on either side -> cosine 0, no NaN."""
    model = _tiny_model()
    grads = [torch.zeros_like(p) for p in model.parameters()]
    names = [n for n, _ in model.named_parameters()]
    result = GradientVerifier(model, _mse_loss).verify(
        grads, names, inputs=torch.zeros(2, 8), labels=torch.zeros(2, 2),
    )
    for c in result.checks:
        assert c.cosine == 0.0
        assert c.norm_ratio == 0.0
