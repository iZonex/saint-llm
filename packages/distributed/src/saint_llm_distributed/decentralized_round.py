"""Decentralized training round — verifier + reputation + aggregator glue.

Single-call orchestrator that ties the trustless-training pieces
together::

    1. Peers submit ``(peer_id, gradient_claim, sample, label)`` tuples.
    2. :class:`GradientVerifier` re-runs forward+backward locally on
       each submission's sample; emits accept/reject.
    3. :class:`PeerReputationTracker` updates each peer's score from
       the verification outcome.
    4. :func:`reputation_weighted_reduce` aggregates the *accepted*
       gradients weighted by current reputation, with the configured
       ``min_score`` floor.

The result is one aggregated gradient list ready to feed into the
outer-loop optimizer (DiLoCo / NoLoCo). Honest peers' contributions
land at full weight; malicious peers are caught by verification on
their first round and lose reputation thereafter — by round ~30 a
fully malicious peer is excluded by the ``min_score`` floor.

No comms here — :class:`PeerSubmission` is in-memory; the real
networking layer (libp2p / TCP / Bittensor) wraps this orchestrator
on the receiving side.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from saint_llm_distributed.reputation import (
    PeerReputationTracker,
    reputation_weighted_reduce,
)
from saint_llm_distributed.verification import (
    GradientVerifier,
    VerificationResult,
)


@dataclass(frozen=True)
class PeerSubmission:
    """One peer's claim for one training round.

    Attributes:
        peer_id:                  unique peer identifier.
        gradient_claim:           list of gradient tensors (one per
            parameter, parallel to ``parameter_names``).
        parameter_names:          parameter names matching the
            verifier model's ``named_parameters()`` keys.
        sample / label:           the (input, label) the peer claims
            to have trained on. The verifier runs forward+backward on
            this exact pair to detect lies.
    """

    peer_id: str
    gradient_claim: list[Tensor]
    parameter_names: list[str]
    sample: Tensor
    label: Tensor


@dataclass(frozen=True)
class PeerOutcome:
    """Per-peer round result emitted by :meth:`DecentralizedTrainingRound.process`."""

    peer_id: str
    accepted: bool
    new_reputation: float
    verification: VerificationResult


@dataclass(frozen=True)
class RoundResult:
    """Aggregate result of a decentralized training round.

    Attributes:
        aggregated:    per-parameter aggregated gradient (tensors are
            in the *parameter-name order of the first accepted peer*;
            see :attr:`parameter_names` for the matching order).
            Empty list when no peers were accepted.
        parameter_names: the parameter-name order matching
            :attr:`aggregated`.
        outcomes:      per-peer accept/reject + new reputation.
        n_accepted:    convenience count.
    """

    aggregated: list[Tensor]
    parameter_names: list[str]
    outcomes: list[PeerOutcome]
    n_accepted: int


class DecentralizedTrainingRound:
    """Single-round orchestrator combining verification + reputation + reduce.

    Args:
        verifier:     a :class:`GradientVerifier`.
        tracker:      a :class:`PeerReputationTracker`. Updated in
            place with each submission's accept/reject outcome.
        min_score:    minimum reputation for a peer's gradient to be
            included in the weighted reduce. Default 0.1.
    """

    def __init__(
        self,
        *,
        verifier: GradientVerifier,
        tracker: PeerReputationTracker,
        min_score: float = 0.1,
    ) -> None:
        self._verifier = verifier
        self._tracker = tracker
        self._min_score = min_score

    def process(
        self, submissions: list[PeerSubmission],
    ) -> RoundResult:
        """Verify, update reputations, aggregate accepted gradients."""
        if not submissions:
            return RoundResult(
                aggregated=[], parameter_names=[],
                outcomes=[], n_accepted=0,
            )

        outcomes: list[PeerOutcome] = []
        accepted_gradients: list[list[Tensor]] = []
        accepted_ids: list[str] = []
        accepted_param_names: list[str] | None = None

        for sub in submissions:
            result = self._verifier.verify(
                sub.gradient_claim,
                sub.parameter_names,
                inputs=sub.sample,
                labels=sub.label,
            )
            new_rep = self._tracker.update(
                sub.peer_id, accepted=result.accepted,
            )
            outcomes.append(PeerOutcome(
                peer_id=sub.peer_id,
                accepted=result.accepted,
                new_reputation=new_rep,
                verification=result,
            ))
            if result.accepted:
                # All accepted peers must agree on parameter order;
                # take the first one's order as canonical.
                if accepted_param_names is None:
                    accepted_param_names = list(sub.parameter_names)
                elif sub.parameter_names != accepted_param_names:
                    raise ValueError(
                        f"peer {sub.peer_id!r} parameter_names diverged "
                        "from earlier accepted peer in this round",
                    )
                accepted_gradients.append(list(sub.gradient_claim))
                accepted_ids.append(sub.peer_id)

        if not accepted_gradients or accepted_param_names is None:
            return RoundResult(
                aggregated=[], parameter_names=[],
                outcomes=outcomes, n_accepted=0,
            )

        aggregated = reputation_weighted_reduce(
            accepted_gradients, accepted_ids, self._tracker,
            min_score=self._min_score,
        )
        return RoundResult(
            aggregated=aggregated,
            parameter_names=accepted_param_names,
            outcomes=outcomes,
            n_accepted=len(accepted_gradients),
        )
