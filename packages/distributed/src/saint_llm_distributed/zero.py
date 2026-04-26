"""Hybrid ZeRO bucket assignment with Muon-aware costing.

Standard ZeRO sharding splits optimizer state across ``N`` ranks so
each rank owns ``1/N`` of the parameters. The packing problem: assign
parameters to buckets to balance per-step **compute cost** across
ranks.

Most optimizers (AdamW, Lion, RMSprop) cost a roughly constant number
of FLOPs per parameter — uniform-size param-count buckets are
optimal. Muon is different: each parameter matrix has its update
computed via Newton-Schulz iteration on the gradient, with cost
proportional to ``n * m`` (matrix dim product) per iteration. A
1024x1024 weight is 1M params but ~1G Newton-Schulz FLOPs; a 128x8192
weight is the same params but ~8G FLOPs — packing by param count
leaves ranks badly imbalanced.

This module ships:

* :func:`pack_buckets_by_cost` — Longest-Processing-Time-first (LPT)
  greedy bucket-packing. Takes a list of per-item costs and a bucket
  count; returns the per-bucket assignment with worst-case ratio of
  ``4/3 - 1/(3N)`` to the optimal makespan (Graham 1969).
* :func:`muon_param_cost` — cost of a single tensor under Muon
  (rounded ``n*m`` for 2D, ``numel()`` for 1D / scalars).
* :func:`adamw_param_cost` — cost = ``numel()`` (AdamW is per-element).
* :func:`hybrid_zero_assign` — runs LPT separately per optimizer
  group then concatenates, so each rank gets balanced workloads from
  both Muon and AdamW pools simultaneously.

The output is a :class:`BucketAssignment` mapping rank index ->
parameter indices. The runtime networking layer (DDP / FSDP2 /
torch.distributed) consumes it and shards optimizer state
accordingly. This module stays free of comm primitives so it can run
unchanged on a single host (tests + offline planning).
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from torch import Tensor


@dataclass(frozen=True)
class BucketAssignment:
    """Result of :func:`pack_buckets_by_cost` / :func:`hybrid_zero_assign`.

    Attributes:
        buckets:        ``buckets[rank]`` is the list of parameter
            indices assigned to that rank, in input order.
        per_rank_cost:  ``per_rank_cost[rank]`` is the summed cost of
            the parameters in that bucket.
    """

    buckets: tuple[tuple[int, ...], ...]
    per_rank_cost: tuple[float, ...]

    @property
    def n_ranks(self) -> int:
        return len(self.buckets)

    @property
    def imbalance(self) -> float:
        """``(max_cost - min_cost) / max(max_cost, eps)``. 0.0 = perfect balance."""
        if not self.per_rank_cost:
            return 0.0
        max_cost = max(self.per_rank_cost)
        if max_cost <= 0.0:
            return 0.0
        return (max_cost - min(self.per_rank_cost)) / max_cost


def pack_buckets_by_cost(
    costs: Sequence[float],
    n_buckets: int,
) -> BucketAssignment:
    """Greedy LPT (Longest Processing Time) bucket packing.

    Sort items by cost descending; for each item, assign to the
    currently-lightest bucket. Worst-case approximation ratio is
    ``4/3 - 1/(3N)`` of the optimal makespan (Graham 1969). For our
    workloads (a few hundred parameter tensors per N <= 64 ranks)
    this is typically within 1-2% of optimal.

    Args:
        costs:     iterable of non-negative per-item costs. Items
            with ``cost <= 0`` are still placed (they don't shift
            the makespan).
        n_buckets: number of buckets / ranks. Must be ``>= 1``.

    Returns:
        :class:`BucketAssignment`. Empty ``costs`` yields ``n_buckets``
        empty lists with zero cost.
    """
    if n_buckets <= 0:
        raise ValueError(f"n_buckets must be positive; got {n_buckets}")
    if any(c < 0.0 for c in costs):
        raise ValueError("costs must be non-negative")

    buckets: list[list[int]] = [[] for _ in range(n_buckets)]
    cumulative = [0.0] * n_buckets
    indexed = sorted(
        range(len(costs)), key=lambda i: float(costs[i]), reverse=True,
    )
    # Min-heap of (cost, rank). Pop the lightest, append, push back.
    heap: list[tuple[float, int]] = [(0.0, r) for r in range(n_buckets)]
    heapq.heapify(heap)
    for idx in indexed:
        c = float(costs[idx])
        light_cost, rank = heapq.heappop(heap)
        buckets[rank].append(idx)
        cumulative[rank] = light_cost + c
        heapq.heappush(heap, (cumulative[rank], rank))

    return BucketAssignment(
        buckets=tuple(tuple(sorted(b)) for b in buckets),
        per_rank_cost=tuple(cumulative),
    )


def muon_param_cost(tensor: Tensor) -> float:
    """Cost approximation for a Muon-managed parameter.

    Newton-Schulz iteration on a ``(n, m)`` matrix runs three
    matmuls per iteration, each ``O(n*m*min(n,m))``. We approximate
    with ``n*m`` for 2D tensors (factoring out the constant
    iteration count + the ``min(n,m)`` factor since they're roughly
    proportional across our weights). 1D + scalar tensors fall back
    to ``numel()`` since Newton-Schulz is skipped (Muon updates them
    via plain SGD).
    """
    if tensor.dim() == 2:
        n, m = tensor.shape
        return float(n * m)
    return float(tensor.numel())


def adamw_param_cost(tensor: Tensor) -> float:
    """Cost approximation for an AdamW-managed parameter.

    AdamW does a constant-FLOP-per-element update (m / v moments +
    a few elementwise ops), so cost = ``numel()``.
    """
    return float(tensor.numel())


def hybrid_zero_assign(
    params: Sequence[Tensor],
    optimizer_kinds: Sequence[Literal["muon", "adamw"]],
    n_buckets: int,
) -> BucketAssignment:
    """Pack parameters into ``n_buckets`` buckets, splitting per optimizer.

    Muon and AdamW are packed independently (each with its own LPT
    run) and the results are concatenated rank-by-rank. The summed
    per-rank cost is the union of both pools, so callers see a
    single balanced view.

    Args:
        params:           list of parameter tensors.
        optimizer_kinds:  parallel list naming the optimizer for each
            parameter ("muon" or "adamw").
        n_buckets:        number of ranks.

    Returns:
        :class:`BucketAssignment` whose ``buckets[rank]`` is the union
        of the rank's Muon picks and AdamW picks (sorted by index).
    """
    if len(params) != len(optimizer_kinds):
        raise ValueError(
            f"params length {len(params)} != optimizer_kinds length "
            f"{len(optimizer_kinds)}",
        )
    if n_buckets <= 0:
        raise ValueError(f"n_buckets must be positive; got {n_buckets}")

    muon_idx = [
        i for i, k in enumerate(optimizer_kinds) if k == "muon"
    ]
    adamw_idx = [
        i for i, k in enumerate(optimizer_kinds) if k == "adamw"
    ]
    others = [
        i for i, k in enumerate(optimizer_kinds)
        if k not in ("muon", "adamw")
    ]
    if others:
        raise ValueError(
            f"optimizer_kinds entries must be 'muon' or 'adamw'; got "
            f"unknown values at indices {others}",
        )

    muon_costs = [muon_param_cost(params[i]) for i in muon_idx]
    adamw_costs = [adamw_param_cost(params[i]) for i in adamw_idx]
    muon_assign = pack_buckets_by_cost(muon_costs, n_buckets)
    adamw_assign = pack_buckets_by_cost(adamw_costs, n_buckets)

    buckets: list[list[int]] = [[] for _ in range(n_buckets)]
    per_rank_cost = [0.0] * n_buckets
    for rank in range(n_buckets):
        for local in muon_assign.buckets[rank]:
            buckets[rank].append(muon_idx[local])
        for local in adamw_assign.buckets[rank]:
            buckets[rank].append(adamw_idx[local])
        per_rank_cost[rank] = (
            muon_assign.per_rank_cost[rank]
            + adamw_assign.per_rank_cost[rank]
        )

    return BucketAssignment(
        buckets=tuple(tuple(sorted(b)) for b in buckets),
        per_rank_cost=tuple(per_rank_cost),
    )
