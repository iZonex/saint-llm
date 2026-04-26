"""NoLoCo — log(N) all-reduce + AdamW outer optimizer.

Gensyn's NoLoCo (2025) extends DiLoCo with two changes that matter for
volunteer-GPU training:

* **log(N) ring all-reduce.** Instead of every worker exchanging
  pseudo-gradients with every other worker (O(N²) bandwidth), each
  worker only talks to its two ring neighbors per round; ``ceil(log2
  N)`` rounds suffice to converge on a global mean. This drops sync
  bandwidth from O(N) per worker to O(log N) and tolerates
  heterogeneous link speeds — slow nodes only block their immediate
  neighbors.
* **AdamW outer optimizer.** DiLoCo's original recipe used SGD-with-
  Nesterov for the outer step; NoLoCo found AdamW more robust to
  noisy reductions over flaky networks.

This module ships:

* :func:`ring_all_reduce` — pure function that computes the reduced
  tensors for every peer given the full set of inputs. The actual
  ring topology is collapsed into a single deterministic computation
  here so unit tests don't need real comms.
* :class:`RingAllReduceSimulator` — multi-peer harness that issues
  a :class:`PeerReducer` per peer; each peer calls
  :meth:`PeerReducer.reduce` and the simulator runs the ring once all
  peers have submitted.
* :class:`NoLoCoConfig` / :class:`NoLoCo` — driver mirroring
  :class:`DiLoCo`'s API but using AdamW for the outer step. Plug in
  any :class:`PseudoGradientReducer` (the simulator's
  :class:`PeerReducer` for tests, a torch.distributed-backed reducer
  in production).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from saint_llm_distributed.diloco import PseudoGradientReducer


def ring_log2_rounds(n_peers: int) -> int:
    """``ceil(log2(n_peers))`` rounds, with 0 for n_peers <= 1."""
    if n_peers <= 1:
        return 0
    return max(1, math.ceil(math.log2(n_peers)))


def ring_all_reduce(
    inputs_per_peer: list[list[Tensor]],
) -> list[list[Tensor]]:
    """Recursive-doubling all-reduce over ``log2(N)`` rounds.

    Round ``k`` exchanges each peer ``i`` with the peer at distance
    ``2**k`` (XOR for power-of-two ``N``). After ``ceil(log2 N)``
    rounds every peer holds the global mean (exact for power-of-two
    N within FP tolerance; small drift for other N because the
    doubling pattern doesn't cleanly tile).

    The implementation is pure-tensor — no comms — so the function
    doubles as the ground-truth reference for any networked
    implementation.

    Args:
        inputs_per_peer: list of length ``N``; each entry is one peer's
            list of pseudo-grad tensors (same parameter ordering across
            peers).

    Returns:
        list of length ``N`` of reduced tensor lists.
    """
    n = len(inputs_per_peer)
    if n == 0:
        return []
    if n == 1:
        return [list(inputs_per_peer[0])]

    state = [
        [t.detach().clone() for t in peer_tensors]
        for peer_tensors in inputs_per_peer
    ]
    is_pow2 = (n & (n - 1)) == 0
    rounds = ring_log2_rounds(n)
    for k in range(rounds):
        distance = 1 << k
        new_state: list[list[Tensor]] = []
        if is_pow2:
            # XOR pairs: pair (i, i ^ distance) and average symmetrically.
            for i in range(n):
                partner = i ^ distance
                me = state[i]
                them = state[partner]
                new_state.append([
                    (a + b) / 2 for a, b in zip(me, them, strict=True)
                ])
        else:
            # Non-power-of-two: pair with peer at +distance modulo n.
            # We average symmetrically so both ends of a pair end up
            # with the same value; over log2(N) rounds this still
            # diffuses widely though without exact-mean guarantee.
            for i in range(n):
                partner = (i + distance) % n
                me = state[i]
                them = state[partner]
                new_state.append([
                    (a + b) / 2 for a, b in zip(me, them, strict=True)
                ])
        state = new_state
    return state


class RingAllReduceSimulator:
    """In-process harness that wires N peers into a ring.

    Each call to :meth:`peer_reducer` returns a fresh
    :class:`PeerReducer` for one peer. Peers call ``reduce(tensors)``
    on their reducer; the simulator buffers inputs, blocks the caller
    on a barrier-style :class:`threading.Condition`, and once all N
    peers have submitted runs the ring computation and wakes everyone
    with their reduced result.

    Two ways to drive it:

    * **Multi-threaded** — start one thread per peer that calls
      ``reduce``. The simulator's barrier handles synchronization. This
      mirrors how a real distributed cluster would call the reducer.
    * **Single-threaded** — for unit tests where you only want to
      verify the math, call :meth:`run_round` directly with all peers'
      tensors and read each peer's reduced result via :meth:`output`.
    """

    def __init__(self, n_peers: int) -> None:
        if n_peers <= 0:
            raise ValueError(f"n_peers must be positive; got {n_peers}")
        self._n = n_peers
        self._cond = threading.Condition()
        self._round_inputs: list[list[Tensor] | None] = [None] * n_peers
        self._round_outputs: list[list[Tensor] | None] = [None] * n_peers
        self._round_id = 0
        self._round_consumed = [True] * n_peers

    @property
    def n_peers(self) -> int:
        return self._n

    def peer_reducer(self, peer_index: int) -> PeerReducer:
        if not 0 <= peer_index < self._n:
            raise ValueError(
                f"peer_index {peer_index} out of range for n_peers={self._n}",
            )
        return PeerReducer(self, peer_index)

    def _submit(self, peer_index: int, tensors: list[Tensor]) -> list[Tensor]:
        with self._cond:
            self._round_inputs[peer_index] = list(tensors)
            self._round_consumed[peer_index] = False
            self._cond.notify_all()
            while any(x is None for x in self._round_inputs):
                self._cond.wait()
            if self._round_outputs[peer_index] is None:
                # First-arriver-after-quorum runs the reduction.
                if all(o is None for o in self._round_outputs):
                    inputs: list[list[Tensor]] = [
                        t for t in self._round_inputs if t is not None
                    ]
                    outputs = ring_all_reduce(inputs)
                    for i in range(self._n):
                        self._round_outputs[i] = outputs[i]
                    self._cond.notify_all()
                else:
                    while self._round_outputs[peer_index] is None:
                        self._cond.wait()
            result = self._round_outputs[peer_index]
            assert result is not None
            self._round_consumed[peer_index] = True
            # Last consumer resets the round so the simulator is
            # ready for another sync cycle.
            if all(self._round_consumed):
                self._round_inputs = [None] * self._n
                self._round_outputs = [None] * self._n
                self._round_id += 1
                self._cond.notify_all()
            return result

    def run_round(
        self, inputs_per_peer: list[list[Tensor]],
    ) -> list[list[Tensor]]:
        """Single-threaded helper: run one ring round with all inputs at once."""
        if len(inputs_per_peer) != self._n:
            raise ValueError(
                f"expected {self._n} peer inputs, got {len(inputs_per_peer)}",
            )
        return ring_all_reduce(inputs_per_peer)


class PeerReducer:
    """One peer's view of the ring; satisfies the Protocol."""

    def __init__(self, simulator: RingAllReduceSimulator, peer_index: int) -> None:
        self._sim = simulator
        self._idx = peer_index

    def reduce(self, tensors: list[Tensor]) -> list[Tensor]:
        return self._sim._submit(self._idx, tensors)


@dataclass
class NoLoCoConfig:
    """NoLoCo outer-loop knobs.

    Attributes:
        inner_steps:        local steps between syncs (typically 500-1500).
        outer_lr:           AdamW learning rate for the outer step.
        outer_beta1:        AdamW first-moment decay.
        outer_beta2:        AdamW second-moment decay.
        outer_eps:          AdamW epsilon.
        outer_weight_decay: AdamW weight decay applied to params each step.
    """

    inner_steps: int = 500
    outer_lr: float = 1e-3
    outer_beta1: float = 0.9
    outer_beta2: float = 0.95
    outer_eps: float = 1e-8
    outer_weight_decay: float = 0.0


class NoLoCo:
    """NoLoCo outer-loop driver.

    Mirrors :class:`DiLoCo`'s API — snapshot + ``inner_step()`` +
    ``outer_sync()`` — but the outer optimizer is AdamW. The reducer is
    typically a :class:`PeerReducer` for tests or a torch.distributed
    ring reducer in production.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        cfg: NoLoCoConfig,
        *,
        reducer: PseudoGradientReducer,
    ) -> None:
        self.cfg = cfg
        self._params = list(params)
        self._reducer = reducer
        self._snapshot: list[Tensor] = [p.detach().clone() for p in self._params]
        self._m: list[Tensor] = [torch.zeros_like(p) for p in self._params]
        self._v: list[Tensor] = [torch.zeros_like(p) for p in self._params]
        self._inner_count = 0
        self._step = 0

    def inner_step(self) -> None:
        self._inner_count += 1

    @property
    def should_sync(self) -> bool:
        return self._inner_count >= self.cfg.inner_steps

    @torch.no_grad()
    def outer_sync(self) -> None:
        """One ring all-reduce + AdamW outer step."""
        pseudo_grads = [
            (self._snapshot[i] - p.detach()) for i, p in enumerate(self._params)
        ]
        reduced = self._reducer.reduce(pseudo_grads)

        self._step += 1
        bias_c1 = 1.0 - self.cfg.outer_beta1**self._step
        bias_c2 = 1.0 - self.cfg.outer_beta2**self._step
        lr_t = self.cfg.outer_lr * math.sqrt(bias_c2) / bias_c1

        for i, p in enumerate(self._params):
            g = reduced[i]
            self._m[i].mul_(self.cfg.outer_beta1).add_(
                g, alpha=1 - self.cfg.outer_beta1,
            )
            self._v[i].mul_(self.cfg.outer_beta2).addcmul_(
                g, g, value=1 - self.cfg.outer_beta2,
            )
            update = self._m[i] / (self._v[i].sqrt() + self.cfg.outer_eps)
            new_param = self._snapshot[i] - lr_t * update
            if self.cfg.outer_weight_decay > 0.0:
                new_param = new_param * (
                    1.0 - self.cfg.outer_lr * self.cfg.outer_weight_decay
                )
            p.data.copy_(new_param)
            self._snapshot[i] = new_param.detach().clone()

        self._inner_count = 0

    def state_dict(self) -> dict[str, object]:
        return {
            "snapshot": [s.detach().clone() for s in self._snapshot],
            "m": [m.detach().clone() for m in self._m],
            "v": [v.detach().clone() for v in self._v],
            "inner_count": self._inner_count,
            "step": self._step,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        snap = state["snapshot"]
        m = state["m"]
        v = state["v"]
        if not isinstance(snap, list) or not isinstance(m, list) or not isinstance(v, list):
            raise TypeError("NoLoCo state_dict has wrong shape")
        for i, s in enumerate(snap):
            self._snapshot[i] = s  # type: ignore[assignment]
        for i, mm in enumerate(m):
            self._m[i] = mm  # type: ignore[assignment]
        for i, vv in enumerate(v):
            self._v[i] = vv  # type: ignore[assignment]
        self._inner_count = int(state.get("inner_count", 0))  # type: ignore[arg-type]
        self._step = int(state.get("step", 0))  # type: ignore[arg-type]
