"""Tests for the decentralized training round driver."""

from __future__ import annotations

import pytest
import torch
from saint_llm_distributed import (
    GradientVerifier,
    PeerReputationConfig,
    PeerReputationTracker,
)
from saint_llm_distributed.decentralized_round import (
    DecentralizedTrainingRound,
    PeerSubmission,
    RoundResult,
)
from torch import nn


def _tiny_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))


def _mse_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((model(x) - y) ** 2).mean()


def _compute_honest_gradient(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
) -> tuple[list[torch.Tensor], list[str]]:
    """Helper: run backward locally on (x, y) to get the honest gradient."""
    for p in model.parameters():
        p.grad = None
    _mse_loss(model, x, y).backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]  # type: ignore[union-attr]
    names = [n for n, _ in model.named_parameters()]
    return grads, names


def _make_round(
    *,
    cosine_threshold: float = 0.99,
    min_score: float = 0.1,
    initial_score: float = 0.5,
    decay: float = 0.9,
) -> tuple[DecentralizedTrainingRound, PeerReputationTracker, nn.Module]:
    model = _tiny_model()
    verifier = GradientVerifier(
        model, _mse_loss, cosine_threshold=cosine_threshold,
    )
    tracker = PeerReputationTracker(PeerReputationConfig(
        initial_score=initial_score, decay=decay,
    ))
    rd = DecentralizedTrainingRound(
        verifier=verifier, tracker=tracker, min_score=min_score,
    )
    return rd, tracker, model


def test_empty_submissions_returns_empty_round_result() -> None:
    rd, _, _ = _make_round()
    out = rd.process([])
    assert isinstance(out, RoundResult)
    assert out.aggregated == []
    assert out.outcomes == []
    assert out.n_accepted == 0


def test_honest_cluster_aggregates_to_mean() -> None:
    """Two honest peers -> aggregated gradient is their mean."""
    rd, _, model = _make_round(initial_score=0.5)
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)

    # Two honest peers — same gradient (since they trained on the same x, y
    # under the same model).
    peer_model = _tiny_model()
    peer_model.load_state_dict(model.state_dict())
    grads, names = _compute_honest_gradient(peer_model, x, y)

    submissions = [
        PeerSubmission(
            peer_id=f"peer{i}",
            gradient_claim=[g.clone() for g in grads],
            parameter_names=list(names),
            sample=x, label=y,
        )
        for i in range(2)
    ]
    out = rd.process(submissions)
    assert out.n_accepted == 2
    # Both reputations should bump up.
    for outcome in out.outcomes:
        assert outcome.accepted is True
        assert outcome.new_reputation > 0.5
    # Aggregated equals each peer's gradient (they're identical).
    for agg, expected in zip(out.aggregated, grads, strict=True):
        assert torch.allclose(agg, expected, atol=1e-5)


def test_malicious_peer_excluded_from_aggregation() -> None:
    """A peer claiming a wrong gradient gets rejected; result follows honest peer."""
    rd, _, model = _make_round()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)

    peer_model = _tiny_model()
    peer_model.load_state_dict(model.state_dict())
    honest_grads, names = _compute_honest_gradient(peer_model, x, y)
    malicious_grads = [torch.randn_like(g) for g in honest_grads]

    out = rd.process([
        PeerSubmission(
            peer_id="alice", gradient_claim=honest_grads,
            parameter_names=list(names), sample=x, label=y,
        ),
        PeerSubmission(
            peer_id="bob", gradient_claim=malicious_grads,
            parameter_names=list(names), sample=x, label=y,
        ),
    ])
    assert out.n_accepted == 1
    assert {o.peer_id: o.accepted for o in out.outcomes} == {
        "alice": True, "bob": False,
    }
    # Aggregated == honest gradient (bob excluded).
    for agg, expected in zip(out.aggregated, honest_grads, strict=True):
        assert torch.allclose(agg, expected, atol=1e-5)


def test_reputations_persist_across_rounds() -> None:
    """Bob's reputation drops over multiple rounds of malicious behavior."""
    rd, tracker, model = _make_round(decay=0.9)
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)

    peer_model = _tiny_model()
    peer_model.load_state_dict(model.state_dict())
    honest_grads, names = _compute_honest_gradient(peer_model, x, y)

    # Run 30 rounds with bob always cheating.
    for _ in range(30):
        rd.process([
            PeerSubmission(
                peer_id="alice", gradient_claim=[g.clone() for g in honest_grads],
                parameter_names=list(names), sample=x, label=y,
            ),
            PeerSubmission(
                peer_id="bob", gradient_claim=[torch.randn_like(g) for g in honest_grads],
                parameter_names=list(names), sample=x, label=y,
            ),
        ])
    assert tracker.score("alice") > 0.95
    assert tracker.score("bob") < 0.05


def test_all_rejected_returns_empty_aggregation() -> None:
    """If every peer is rejected (random gradients), aggregated is empty."""
    rd, _, model = _make_round()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    # Use shapes matching the model so verification only fails on cosine.
    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    out = rd.process([
        PeerSubmission(
            peer_id=f"p{i}",
            gradient_claim=[torch.randn(s) * 100.0 for s in shapes],
            parameter_names=list(names),
            sample=x, label=y,
        )
        for i in range(2)
    ])
    assert out.n_accepted == 0
    assert out.aggregated == []
    assert out.parameter_names == []
    # All outcomes still recorded, with rejected accept flags.
    assert all(o.accepted is False for o in out.outcomes)


def test_param_name_divergence_raises() -> None:
    """Two accepted peers with different parameter_name orders -> error."""
    rd, _, model = _make_round(cosine_threshold=0.99)
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)

    peer_model = _tiny_model()
    peer_model.load_state_dict(model.state_dict())
    honest_grads, names = _compute_honest_gradient(peer_model, x, y)

    # Reverse one peer's parameter order so it diverges from the canonical.
    reversed_names = list(reversed(names))
    reversed_grads = list(reversed(honest_grads))

    with pytest.raises(ValueError, match="parameter_names diverged"):
        rd.process([
            PeerSubmission(
                peer_id="alice", gradient_claim=honest_grads,
                parameter_names=list(names), sample=x, label=y,
            ),
            PeerSubmission(
                peer_id="bob", gradient_claim=reversed_grads,
                parameter_names=reversed_names, sample=x, label=y,
            ),
        ])


def test_outcomes_carry_verification_details() -> None:
    rd, _, model = _make_round()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    peer_model = _tiny_model()
    peer_model.load_state_dict(model.state_dict())
    grads, names = _compute_honest_gradient(peer_model, x, y)

    out = rd.process([
        PeerSubmission(
            peer_id="alice", gradient_claim=grads,
            parameter_names=list(names), sample=x, label=y,
        ),
    ])
    outcome = out.outcomes[0]
    assert outcome.peer_id == "alice"
    assert outcome.verification.accepted is True
    assert outcome.verification.mean_cosine > 0.99
    assert "accepted" in outcome.verification.reason


def test_aggregated_gradient_shape_matches_claim() -> None:
    rd, _, model = _make_round()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    peer_model = _tiny_model()
    peer_model.load_state_dict(model.state_dict())
    grads, names = _compute_honest_gradient(peer_model, x, y)

    out = rd.process([
        PeerSubmission(
            peer_id="alice", gradient_claim=grads,
            parameter_names=list(names), sample=x, label=y,
        ),
    ])
    for agg, claim in zip(out.aggregated, grads, strict=True):
        assert agg.shape == claim.shape


def test_round_result_n_accepted_matches_outcomes() -> None:
    rd, _, model = _make_round()
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    peer_model = _tiny_model()
    peer_model.load_state_dict(model.state_dict())
    honest, names = _compute_honest_gradient(peer_model, x, y)
    out = rd.process([
        PeerSubmission("p1", honest, list(names), x, y),
        PeerSubmission("p2", [torch.randn_like(g) for g in honest], list(names), x, y),
        PeerSubmission("p3", [g.clone() for g in honest], list(names), x, y),
    ])
    accepted_count = sum(1 for o in out.outcomes if o.accepted)
    assert out.n_accepted == accepted_count
    # Two honest, one malicious -> 2 accepts.
    assert accepted_count == 2


def test_reputations_updated_for_every_peer_regardless_of_accept() -> None:
    """Accepted and rejected peers all get reputation updates."""
    rd, tracker, model = _make_round(decay=0.5)
    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    peer_model = _tiny_model()
    peer_model.load_state_dict(model.state_dict())
    honest, names = _compute_honest_gradient(peer_model, x, y)

    rd.process([
        PeerSubmission("alice", honest, list(names), x, y),
        PeerSubmission("bob", [torch.randn_like(g) for g in honest], list(names), x, y),
    ])
    # Alice accepted -> score went up from 0.5; bob rejected -> went down.
    assert tracker.score("alice") > 0.5
    assert tracker.score("bob") < 0.5
