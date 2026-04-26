"""Tests for peer reputation tracker + weighted aggregator."""

from __future__ import annotations

import pytest
import torch
from saint_llm_distributed.reputation import (
    PeerReputationConfig,
    PeerReputationTracker,
    reputation_weighted_reduce,
)


def test_initial_score_for_unseen_peer() -> None:
    tracker = PeerReputationTracker(PeerReputationConfig(initial_score=0.5))
    assert tracker.score("alice") == 0.5


def test_accept_pulls_score_toward_one() -> None:
    cfg = PeerReputationConfig(initial_score=0.5, decay=0.9)
    tracker = PeerReputationTracker(cfg)
    new = tracker.update("alice", accepted=True)
    # 0.9 * 0.5 + 0.1 * 1.0 = 0.55
    assert pytest.approx(new, abs=1e-6) == 0.55
    assert tracker.score("alice") == new


def test_reject_pulls_score_toward_zero() -> None:
    cfg = PeerReputationConfig(initial_score=0.5, decay=0.9)
    tracker = PeerReputationTracker(cfg)
    new = tracker.update("alice", accepted=False)
    # 0.9 * 0.5 + 0.1 * 0.0 = 0.45
    assert pytest.approx(new, abs=1e-6) == 0.45


def test_repeated_accepts_converge_toward_one() -> None:
    tracker = PeerReputationTracker(PeerReputationConfig(initial_score=0.5))
    for _ in range(100):
        tracker.update("alice", accepted=True)
    assert tracker.score("alice") > 0.99


def test_repeated_rejects_converge_toward_zero() -> None:
    tracker = PeerReputationTracker(PeerReputationConfig(initial_score=0.5))
    for _ in range(100):
        tracker.update("alice", accepted=False)
    assert tracker.score("alice") < 0.01


def test_score_clamped_to_floor_and_ceiling() -> None:
    cfg = PeerReputationConfig(
        initial_score=0.5, decay=0.0, floor=0.1, ceiling=0.9,
    )
    tracker = PeerReputationTracker(cfg)
    # Single accept with decay=0 jumps directly to target=1.0, then clamped.
    new_high = tracker.update("alice", accepted=True)
    assert new_high == 0.9
    new_low = tracker.update("bob", accepted=False)
    assert new_low == 0.1


def test_update_batch_processes_in_order() -> None:
    tracker = PeerReputationTracker(PeerReputationConfig(initial_score=0.5))
    scores = tracker.update_batch(
        ["a", "b", "c"], [True, False, True],
    )
    assert len(scores) == 3
    assert scores[0] > 0.5  # accepted
    assert scores[1] < 0.5  # rejected
    assert scores[2] > 0.5  # accepted


def test_update_batch_length_mismatch_raises() -> None:
    tracker = PeerReputationTracker()
    with pytest.raises(ValueError, match="length"):
        tracker.update_batch(["a", "b"], [True])


def test_remove_drops_peer() -> None:
    tracker = PeerReputationTracker(PeerReputationConfig(initial_score=0.3))
    tracker.update("alice", accepted=True)
    assert "alice" in tracker.all_scores()
    assert tracker.remove("alice") is True
    # Removing again returns False.
    assert tracker.remove("alice") is False
    # Score reverts to initial for removed peer.
    assert tracker.score("alice") == 0.3


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="initial_score"):
        PeerReputationConfig(initial_score=1.5)
    with pytest.raises(ValueError, match="decay"):
        PeerReputationConfig(decay=1.0)
    with pytest.raises(ValueError, match="floor"):
        PeerReputationConfig(floor=0.7, ceiling=0.3)


# ---- reputation_weighted_reduce -------------------------------------


def test_weighted_reduce_uniform_when_all_scores_equal() -> None:
    """All peers same reputation -> uniform mean across them."""
    tracker = PeerReputationTracker(PeerReputationConfig(initial_score=0.5))
    grads = [
        [torch.tensor([2.0, 4.0])],
        [torch.tensor([0.0, 8.0])],
    ]
    out = reputation_weighted_reduce(grads, ["a", "b"], tracker)
    assert torch.allclose(out[0], torch.tensor([1.0, 6.0]))


def test_weighted_reduce_weighting_skews_toward_high_reputation() -> None:
    """Higher reputation -> more weight in the average."""
    tracker = PeerReputationTracker(PeerReputationConfig(initial_score=0.5))
    # Train Alice up; train Bob down.
    for _ in range(50):
        tracker.update("alice", accepted=True)
        tracker.update("bob", accepted=False)

    # Alice claims [10, 10]; Bob claims [0, 0].
    grads = [
        [torch.tensor([10.0, 10.0])],
        [torch.tensor([0.0, 0.0])],
    ]
    out = reputation_weighted_reduce(grads, ["alice", "bob"], tracker)
    # Should be much closer to Alice's claim than the uniform 5.0.
    assert out[0][0].item() > 9.0


def test_weighted_reduce_min_score_excludes_low_peers() -> None:
    """Peers below min_score get zero weight."""
    tracker = PeerReputationTracker(PeerReputationConfig(initial_score=0.5))
    for _ in range(100):
        tracker.update("bob", accepted=False)  # bob crashes to 0
    # alice stays at initial 0.5; bob is ~0.
    grads = [
        [torch.tensor([10.0])],   # alice
        [torch.tensor([100.0])],  # bob — should be excluded
    ]
    out = reputation_weighted_reduce(
        grads, ["alice", "bob"], tracker, min_score=0.1,
    )
    # Only alice contributes -> result == alice's gradient.
    assert torch.allclose(out[0], torch.tensor([10.0]))


def test_weighted_reduce_falls_back_to_uniform_when_all_excluded() -> None:
    """If every peer is below min_score, fall back to uniform mean."""
    tracker = PeerReputationTracker(PeerReputationConfig(initial_score=0.0))
    grads = [
        [torch.tensor([2.0])],
        [torch.tensor([4.0])],
    ]
    # min_score=0.5 with everyone at 0.0 -> nobody eligible -> uniform.
    out = reputation_weighted_reduce(
        grads, ["a", "b"], tracker, min_score=0.5,
    )
    assert torch.allclose(out[0], torch.tensor([3.0]))


def test_weighted_reduce_handles_multiple_params() -> None:
    """Per-parameter aggregation across peers."""
    tracker = PeerReputationTracker(PeerReputationConfig(initial_score=0.5))
    grads = [
        [torch.tensor([1.0]), torch.tensor([10.0, 20.0])],
        [torch.tensor([3.0]), torch.tensor([30.0, 40.0])],
    ]
    out = reputation_weighted_reduce(grads, ["a", "b"], tracker)
    assert torch.allclose(out[0], torch.tensor([2.0]))
    assert torch.allclose(out[1], torch.tensor([20.0, 30.0]))


def test_weighted_reduce_empty_returns_empty() -> None:
    tracker = PeerReputationTracker()
    assert reputation_weighted_reduce([], [], tracker) == []


def test_weighted_reduce_length_mismatch_raises() -> None:
    tracker = PeerReputationTracker()
    with pytest.raises(ValueError, match="length"):
        reputation_weighted_reduce(
            [[torch.zeros(1)]], ["a", "b"], tracker,
        )


def test_weighted_reduce_param_count_mismatch_raises() -> None:
    tracker = PeerReputationTracker()
    grads = [
        [torch.zeros(1), torch.zeros(2)],
        [torch.zeros(1)],  # mismatched
    ]
    with pytest.raises(ValueError, match="same number"):
        reputation_weighted_reduce(grads, ["a", "b"], tracker)


def test_weighted_reduce_integrates_with_verification_outcomes() -> None:
    """End-to-end: verification accept/reject signals drive reputation drive aggregation."""
    tracker = PeerReputationTracker(PeerReputationConfig(initial_score=0.5))
    # Simulate 60 rounds where alice is honest (always accepted) and
    # bob is malicious (always rejected). With decay=0.95, this is
    # enough rounds for bob's score to fall well below 0.1.
    for _ in range(60):
        tracker.update_batch(["alice", "bob"], [True, False])
    grads = [
        [torch.tensor([1.0])],   # alice — honest gradient
        [torch.tensor([99.0])],  # bob — malicious outlier
    ]
    out = reputation_weighted_reduce(
        grads, ["alice", "bob"], tracker, min_score=0.1,
    )
    # Bob should be excluded; result very close to alice's claim.
    assert pytest.approx(out[0].item(), abs=1e-3) == 1.0
