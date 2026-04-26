"""DiLoCo-family outer-loop synchronization.

DiLoCo (Decoupled iLearn-Communicate; Gensyn / Google) decouples
inner SGD steps (fast, local) from outer parameter sync (slow, with
peers). Each worker:

1. Runs ``inner_steps`` local AdamW updates.
2. Computes the local "pseudo-gradient" = old_params - new_params.
3. Synchronizes pseudo-gradients with peers.
4. Applies an outer optimizer step (SGD with Nesterov per the
   original recipe, or AdamW per NoLoCo) using the synchronized
   pseudo-gradients to update parameters.

This module ships the *algorithmic skeleton*. The actual all-reduce /
peer-bus is supplied by the runtime via the
:class:`PseudoGradientReducer` Protocol — production deploys
``torch.distributed.all_reduce`` or a Bittensor / Pluralis comm
layer; tests pass an in-process mock.

References:
    Douillard et al. 2023 (DiLoCo)
    Gensyn NoLoCo (2025) — log(N) all-reduce
    Templar SparseLoCo (Bittensor SN3, March 2026)
    Decoupled DiLoCo (arXiv 2604.21428)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class PseudoGradientReducer(Protocol):
    """Reduce per-parameter pseudo-gradients across peers.

    Production implementations call ``torch.distributed.all_reduce`` or
    a custom networking layer; tests substitute an in-process mock.

    The reducer is given a list of tensors (one per parameter) and
    must return the same-shape list with the cross-peer reduction
    applied (mean by default).
    """

    def reduce(self, tensors: list[Tensor]) -> list[Tensor]: ...


@dataclass
class _MeanReducer:
    """Trivial in-process reducer — used when there's no peer network.

    Returns inputs unchanged so single-node testing produces an
    identical loss curve to a non-DiLoCo run with the same inner-step
    count.
    """

    def reduce(self, tensors: list[Tensor]) -> list[Tensor]:
        return tensors


@dataclass
class DiLoCoConfig:
    """Outer-loop knobs.

    Attributes:
        inner_steps:    local SGD/Adam steps between syncs. The
            published DiLoCo recipe uses 100-500. NoLoCo / SparseLoCo
            often go higher (1000+).
        outer_lr:       learning rate for the outer SGD step that
            applies the synced pseudo-gradient. 0.7 in the original
            DiLoCo paper.
        outer_momentum: Nesterov momentum for the outer step. 0.9 is
            standard.
        nesterov:       whether to use Nesterov momentum. Default True.
    """

    inner_steps: int = 100
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    nesterov: bool = True


class DiLoCo:
    """Stateful DiLoCo outer-loop driver.

    Wrap your model + inner optimizer; ``inner_step()`` for each local
    SGD step; ``outer_sync()`` after ``cfg.inner_steps`` to do one
    cross-peer sync + outer optimizer step.

    The driver maintains the snapshot of parameters at the start of
    the inner loop so the pseudo-gradient ``snapshot - current`` is
    the right thing to send to peers. Outer momentum buffer per
    parameter is held internally.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        cfg: DiLoCoConfig,
        *,
        reducer: PseudoGradientReducer | None = None,
    ) -> None:
        self.cfg = cfg
        self._params = list(params)
        self._reducer = reducer if reducer is not None else _MeanReducer()
        self._snapshot: list[Tensor] = [p.detach().clone() for p in self._params]
        self._momentum_buffer: list[Tensor] = [
            torch.zeros_like(p) for p in self._params
        ]
        self._inner_count = 0

    def inner_step(self) -> None:
        """Increment the inner-step counter. Call after each local optimizer.step()."""
        self._inner_count += 1

    @property
    def should_sync(self) -> bool:
        return self._inner_count >= self.cfg.inner_steps

    @torch.no_grad()
    def outer_sync(self) -> None:
        """One cross-peer sync + outer optimizer step.

        Computes pseudo-gradients ``snapshot - current``, reduces
        across peers (mean), applies SGD-with-Nesterov outer update to
        params using ``cfg.outer_lr`` and ``cfg.outer_momentum``,
        then re-snapshots for the next inner round.
        """
        # Pseudo-gradient = old - new = snapshot - current.
        pseudo_grads = [
            (self._snapshot[i] - p.detach()) for i, p in enumerate(self._params)
        ]
        # Reduce across peers (mean by convention; reducer can override).
        reduced = self._reducer.reduce(pseudo_grads)

        # Outer SGD-with-Nesterov-momentum update.
        for i, p in enumerate(self._params):
            g = reduced[i]
            buf = self._momentum_buffer[i]
            buf.mul_(self.cfg.outer_momentum).add_(g)
            update = buf if not self.cfg.nesterov else (g + self.cfg.outer_momentum * buf)
            # Param update: param = snapshot - outer_lr * update.
            new_param = self._snapshot[i] - self.cfg.outer_lr * update
            p.data.copy_(new_param)
            # Re-snapshot for next inner round.
            self._snapshot[i] = new_param.detach().clone()

        self._inner_count = 0

    def state_dict(self) -> dict[str, object]:
        return {
            "snapshot": [s.detach().clone() for s in self._snapshot],
            "momentum_buffer": [m.detach().clone() for m in self._momentum_buffer],
            "inner_count": self._inner_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        snap = state["snapshot"]
        mom = state["momentum_buffer"]
        if not isinstance(snap, list) or not isinstance(mom, list):
            raise TypeError("DiLoCo state_dict has wrong shape")
        for i, s in enumerate(snap):
            self._snapshot[i] = s  # type: ignore[assignment]
        for i, m in enumerate(mom):
            self._momentum_buffer[i] = m  # type: ignore[assignment]
        self._inner_count = int(state.get("inner_count", 0))  # type: ignore[arg-type]
