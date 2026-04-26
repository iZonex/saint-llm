"""Gradient verification — trustless re-computation for decentralized training.

In volunteer-GPU / Bitcoin-style training, peers compute gradients on
their local data and submit them to the cluster. Without trust, a
malicious peer can submit *anything* — random noise, an inverted
gradient, last week's update — and the aggregator has no way to know.
The cluster either drops to a permissioned set (defeats the BTC
vision) or adds a verification protocol.

This module implements the simplest viable verifier:

* :class:`GradientVerifier` — on receiving ``(peer_gradient,
  sample, label)``, re-runs forward + backward locally on the same
  ``(sample, label)`` and computes per-parameter cosine similarity
  against the peer's claim. Returns an accept/reject decision based
  on a configurable threshold.

The cost is a single extra backward pass on the verifier's side. In
practice you sub-sample the peer's batch (e.g. 1 in 100 examples) so
the overhead is small. The protocol assumes:

* Peer + verifier hold the same model checkpoint at the start of the
  contested step.
* The training batch is reproducible from a peer-supplied seed (or
  the peer ships the actual sample).

These assumptions match standard parameter-server / DiLoCo flows.

References:
    Bittensor SN3 / Templar verification design (March 2026)
    Gensyn Verde verification game (2024)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

LossFn = Callable[[torch.nn.Module, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class GradientCheck:
    """Result of comparing one parameter's claimed gradient vs local."""

    name: str
    cosine: float
    norm_ratio: float

    @property
    def passes(self) -> bool:
        """True iff cosine is finite and positive (sub-threshold checks belong to the verifier)."""
        return torch.isfinite(torch.tensor(self.cosine)).item() and self.cosine > 0.0


@dataclass(frozen=True)
class VerificationResult:
    """Per-parameter checks + aggregate accept decision."""

    accepted: bool
    mean_cosine: float
    min_cosine: float
    checks: tuple[GradientCheck, ...]

    @property
    def reason(self) -> str:
        if self.accepted:
            return "accepted"
        if self.min_cosine <= 0.0:
            return f"rejected: min_cosine={self.min_cosine:.3f} <= 0"
        return f"rejected: mean_cosine={self.mean_cosine:.3f} below threshold"


class GradientVerifier:
    """Verify peer-submitted gradients by re-computing locally.

    Args:
        model:                the *verifier's* copy of the model.
            Must hold the same parameters as the peer at the start
            of the contested step.
        loss_fn:              callable
            ``(model, inputs, labels) -> scalar_loss``.
        cosine_threshold:     accept when ``mean(per_param_cosine)
            >= threshold`` and every per-param cosine is positive.
            Default 0.95.
        param_filter:         optional ``(name) -> bool`` filter
            picking which params count toward verification (e.g.
            skip embeddings to dodge token-frequency noise).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: LossFn,
        *,
        cosine_threshold: float = 0.95,
        param_filter: Callable[[str], bool] | None = None,
    ) -> None:
        if not 0.0 <= cosine_threshold <= 1.0:
            raise ValueError(
                f"cosine_threshold must be in [0, 1]; got {cosine_threshold}",
            )
        self._model = model
        self._loss_fn = loss_fn
        self._threshold = cosine_threshold
        self._filter = param_filter

    def verify(
        self,
        peer_gradients: Iterable[Tensor],
        peer_param_names: Sequence[str],
        *,
        inputs: Tensor,
        labels: Tensor,
    ) -> VerificationResult:
        """Compare peer's claim against a fresh local backward pass.

        Args:
            peer_gradients:  iterable of gradient tensors (one per
                parameter) parallel to ``peer_param_names``.
            peer_param_names: parameter names matching the model's
                ``named_parameters()`` keys. Names not present in the
                model are silently skipped.
            inputs / labels:  the same training batch the peer used.
                The verifier re-runs forward+backward on this.
        """
        peer_grads_list = list(peer_gradients)
        if len(peer_grads_list) != len(peer_param_names):
            raise ValueError(
                f"peer_gradients length {len(peer_grads_list)} != "
                f"peer_param_names length {len(peer_param_names)}",
            )

        local_grads = _local_backward(self._model, self._loss_fn, inputs, labels)
        checks: list[GradientCheck] = []
        for name, claim in zip(peer_param_names, peer_grads_list, strict=True):
            if name not in local_grads:
                # Peer claims a param the verifier doesn't have — skip.
                continue
            if self._filter is not None and not self._filter(name):
                continue
            local = local_grads[name]
            if claim.shape != local.shape:
                raise ValueError(
                    f"shape mismatch on {name!r}: peer={tuple(claim.shape)} "
                    f"vs local={tuple(local.shape)}",
                )
            cos, norm_ratio = _cosine_and_norm(claim, local)
            checks.append(GradientCheck(name=name, cosine=cos, norm_ratio=norm_ratio))

        if not checks:
            # Nothing to verify against — refuse to accept.
            return VerificationResult(
                accepted=False, mean_cosine=0.0, min_cosine=0.0,
                checks=(),
            )

        cos_values = [c.cosine for c in checks]
        mean_cos = sum(cos_values) / len(cos_values)
        min_cos = min(cos_values)
        accepted = (
            min_cos > 0.0
            and mean_cos >= self._threshold
            and all(torch.isfinite(torch.tensor(c)).item() for c in cos_values)
        )
        return VerificationResult(
            accepted=accepted, mean_cosine=mean_cos, min_cosine=min_cos,
            checks=tuple(checks),
        )


def _local_backward(
    model: torch.nn.Module,
    loss_fn: LossFn,
    inputs: Tensor,
    labels: Tensor,
) -> dict[str, Tensor]:
    """Run forward + backward; return a fresh per-parameter gradient dict."""
    # Save and restore existing grads so the verifier doesn't trample
    # whatever the caller had accumulated.
    saved: list[tuple[torch.nn.Parameter, Tensor | None]] = [
        (p, p.grad.detach().clone() if p.grad is not None else None)
        for p in model.parameters()
    ]
    for p in model.parameters():
        p.grad = None

    was_training = model.training
    try:
        model.eval()  # disable dropout for reproducibility across peers
        loss = loss_fn(model, inputs, labels)
        if not isinstance(loss, Tensor):
            raise TypeError(
                f"loss_fn must return a Tensor; got {type(loss).__name__}",
            )
        loss.backward()
        out: dict[str, Tensor] = {
            name: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
            for name, p in model.named_parameters()
        }
    finally:
        model.train(was_training)
        # Restore original grads.
        for p, g in saved:
            p.grad = g
    return out


def _cosine_and_norm(a: Tensor, b: Tensor) -> tuple[float, float]:
    """Cosine similarity + ``norm(a) / max(norm(b), eps)``."""
    a_flat = a.detach().reshape(-1).to(torch.float32)
    b_flat = b.detach().reshape(-1).to(torch.float32)
    a_norm = a_flat.norm()
    b_norm = b_flat.norm()
    if a_norm.item() == 0.0 or b_norm.item() == 0.0:
        return 0.0, 0.0
    cos = float((a_flat @ b_flat) / (a_norm * b_norm))
    return cos, float(a_norm / b_norm)


def random_peer_gradient(
    model: torch.nn.Module,
    *,
    scale: float = 1.0,
    seed: int | None = None,
) -> tuple[list[Tensor], list[str]]:
    """Helper: generate a random "malicious" peer gradient claim.

    Used in tests + adversarial-robustness drills. Returns
    ``(grads, names)`` parallel lists matching the model's
    ``named_parameters()``.
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    grads: list[Tensor] = []
    names: list[str] = []
    for name, p in model.named_parameters():
        grads.append(torch.randn(p.shape, generator=g) * scale)
        names.append(name)
    return grads, names
