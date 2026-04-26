"""Peer reputation — weighted aggregation for trustless decentralized training.

:class:`GradientVerifier` produces per-peer accept/reject signals.
This module turns those signals into a long-running reputation score
and a reputation-weighted aggregator. Together they close the
trustless training loop:

    1. Peer ``p`` submits ``(gradient_claim, sample)``.
    2. :class:`GradientVerifier` checks the claim; emits ``accepted``.
    3. :class:`PeerReputationTracker` updates ``score(p)``.
    4. :func:`reputation_weighted_reduce` aggregates the round's
       gradients weighted by current reputation.

The tracker uses an exponential moving average — a single bad
gradient drops a peer's score by ``(1 - decay)`` toward 0; a single
good gradient pulls it by the same amount toward 1. Default
``decay=0.95`` gives a half-life of ~14 rounds, which matches
empirical Bittensor SN3 reputation half-lives.

Peers below ``min_score`` are excluded from the weighted average; the
aggregator falls back to a uniform mean when *every* peer is below the
floor (degenerate round — caller should re-sample).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class PeerReputationConfig:
    """Reputation tracker knobs.

    Attributes:
        initial_score:  starting score for unseen peers. ``0.5`` is
            "neutral" — neither trusted nor distrusted.
        decay:          EWMA decay coefficient. Higher = slower update.
            Default ``0.95`` (half-life ≈ 14 updates).
        floor:          lower bound on the score. ``0.0`` lets a peer
            fall to zero (effective ban); use a small positive value
            to keep them eligible after re-validation.
        ceiling:        upper bound. Default ``1.0``.
    """

    initial_score: float = 0.5
    decay: float = 0.95
    floor: float = 0.0
    ceiling: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.initial_score <= 1.0:
            raise ValueError(
                f"initial_score must be in [0, 1]; got {self.initial_score}",
            )
        if not 0.0 <= self.decay < 1.0:
            raise ValueError(f"decay must be in [0, 1); got {self.decay}")
        if not 0.0 <= self.floor <= self.ceiling <= 1.0:
            raise ValueError(
                f"need 0 <= floor ({self.floor}) <= ceiling "
                f"({self.ceiling}) <= 1",
            )


class PeerReputationTracker:
    """In-memory reputation index keyed by peer ID."""

    def __init__(self, cfg: PeerReputationConfig | None = None) -> None:
        self._cfg = cfg if cfg is not None else PeerReputationConfig()
        self._scores: dict[str, float] = {}

    @property
    def cfg(self) -> PeerReputationConfig:
        return self._cfg

    def score(self, peer_id: str) -> float:
        """Return the peer's current reputation (initial_score if unseen)."""
        return self._scores.get(peer_id, self._cfg.initial_score)

    def update(self, peer_id: str, *, accepted: bool) -> float:
        """EWMA-update with the latest accept/reject signal; return new score."""
        target = 1.0 if accepted else 0.0
        prev = self._scores.get(peer_id, self._cfg.initial_score)
        new = self._cfg.decay * prev + (1.0 - self._cfg.decay) * target
        new = max(self._cfg.floor, min(self._cfg.ceiling, new))
        self._scores[peer_id] = new
        return new

    def update_batch(
        self,
        peer_ids: Sequence[str],
        accepted: Sequence[bool],
    ) -> list[float]:
        """Update multiple peers in one call; return new scores in input order."""
        if len(peer_ids) != len(accepted):
            raise ValueError(
                f"peer_ids length {len(peer_ids)} != accepted length "
                f"{len(accepted)}",
            )
        return [
            self.update(pid, accepted=ok)
            for pid, ok in zip(peer_ids, accepted, strict=True)
        ]

    def remove(self, peer_id: str) -> bool:
        """Drop a peer entirely. Returns True if they existed."""
        return self._scores.pop(peer_id, None) is not None

    def all_scores(self) -> dict[str, float]:
        return dict(self._scores)


def reputation_weighted_reduce(
    peer_gradients: Sequence[Sequence[Tensor]],
    peer_ids: Sequence[str],
    tracker: PeerReputationTracker,
    *,
    min_score: float = 0.0,
    eps: float = 1e-8,
) -> list[Tensor]:
    """Reputation-weighted mean of per-peer gradient lists.

    Args:
        peer_gradients:  list of length ``N`` peers; each entry is
            their gradient tensor list (one per parameter, parallel
            across peers).
        peer_ids:        identifiers parallel to ``peer_gradients``.
        tracker:         a :class:`PeerReputationTracker`.
        min_score:       peers with reputation strictly below this are
            excluded from the weighted average. ``0.0`` keeps everyone;
            ``0.5`` keeps only peers that are at-least-neutral.
        eps:             numerical floor; if the kept-peer total mass
            falls below ``eps`` the aggregator falls back to a uniform
            average over all peers (degenerate round signal).

    Returns:
        list of reduced tensors (one per parameter), same shape as the
        per-peer gradient tensors.
    """
    if not peer_gradients:
        return []
    if len(peer_gradients) != len(peer_ids):
        raise ValueError(
            f"peer_gradients length {len(peer_gradients)} != peer_ids "
            f"length {len(peer_ids)}",
        )
    n = len(peer_gradients)
    n_params = len(peer_gradients[0])
    if any(len(g) != n_params for g in peer_gradients):
        raise ValueError(
            "every peer must submit the same number of parameter gradients",
        )

    weights = [tracker.score(pid) for pid in peer_ids]
    eligible = [w if w >= min_score else 0.0 for w in weights]
    total = sum(eligible)
    if total < eps:
        # Degenerate round — fall back to uniform mean.
        eligible = [1.0] * n
        total = float(n)

    normalized = [w / total for w in eligible]
    reduced: list[Tensor] = []
    for j in range(n_params):
        first = peer_gradients[0][j]
        acc = first.new_zeros(first.shape)
        for i in range(n):
            if normalized[i] == 0.0:
                continue
            acc = acc + normalized[i] * peer_gradients[i][j]
        reduced.append(acc)
    return reduced
