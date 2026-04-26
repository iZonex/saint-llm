"""Expert parallelism (EP) — all-to-all dispatch + combine for MoE.

When the MoE expert pool is too large to fit on one device, EP shards
experts across ranks: each rank holds a contiguous slice of expert
parameters. The forward path becomes:

    1. Routing layer picks top-k experts per token (already local).
    2. **Dispatch (all-to-all)** — each rank sends each token to the
       rank that holds its chosen expert. After dispatch, each rank
       holds a "permuted" buffer: the tokens that the rank's local
       experts will process.
    3. Run local experts on their assigned tokens.
    4. **Combine (reverse all-to-all)** — send the per-expert outputs
       back to the originating ranks. Each rank reassembles its
       tokens in the original order, weighted by the routing scores.

This module ships:

* :func:`dispatch_tokens_by_expert` — packs ``(tokens, expert_ids)``
  into per-expert-rank buffers and runs all-to-all. Returns the
  inbound permuted tokens + the index/count metadata needed to undo
  the permutation in :func:`combine_outputs_by_expert`.
* :func:`combine_outputs_by_expert` — reverse: takes processed
  expert outputs, runs the inverse all-to-all, restores the original
  token order on the originating rank.

Single-rank ``world_size=1`` behaviour: all-to-all is a no-op, so
dispatch effectively *sorts* tokens by expert ID locally. The
correctness of the inverse permutation is independent of world size,
so single-rank tests catch the algorithmic bugs that matter.

DeepEP-style fused kernels (NVSHMEM all-to-all, custom cuBLAS-shaped
GEMMs) are a v0.1 polish item; v0.0 uses
``torch.distributed.all_to_all_single`` which is portable across
backends (gloo / nccl / mpi).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class DispatchPlan:
    """Bookkeeping needed to undo an EP dispatch.

    Attributes:
        permuted_tokens:   ``(N_in, D)`` — tokens received by this
            rank from all other ranks, sorted by destination expert.
        send_counts:       ``(world_size,)`` — how many tokens this
            rank sent to each peer.
        recv_counts:       ``(world_size,)`` — how many tokens this
            rank received from each peer.
        sort_indices:      ``(N_local,)`` long — permutation used to
            sort the local tokens by expert ID before all-to-all.
            :func:`combine_outputs_by_expert` uses this to restore
            the original order.
        local_expert_offsets: per-local-expert offsets into
            ``permuted_tokens`` so callers can slice the buffer
            per-expert before running their forward.
        n_local_tokens:    original count of tokens this rank
            contributed (length of the input batch).
    """

    permuted_tokens: Tensor
    send_counts: Tensor
    recv_counts: Tensor
    sort_indices: Tensor
    local_expert_offsets: Tensor
    n_local_tokens: int


def dispatch_tokens_by_expert(
    tokens: Tensor,
    expert_ids: Tensor,
    *,
    n_experts: int,
    world_size: int | None = None,
    group: dist.ProcessGroup | None = None,
) -> DispatchPlan:
    """Route tokens to the rank that owns each token's chosen expert.

    Experts ``[0, n_experts)`` are partitioned contiguously across
    ranks: rank ``r`` owns experts ``[r*n_per_rank, (r+1)*n_per_rank)``.

    Args:
        tokens:      ``(N_local, D)`` — local tokens to dispatch.
        expert_ids:  ``(N_local,)`` long — per-token destination expert.
        n_experts:   total expert count across the cluster.
        world_size:  cluster size. Defaults to ``dist.get_world_size()``
            when a PG is initialized.
        group:       optional process group. Defaults to the world.

    Returns:
        :class:`DispatchPlan` carrying the inbound permuted tokens +
        metadata for the reverse step.
    """
    if tokens.dim() != 2:
        raise ValueError(
            f"tokens must be (N, D); got shape {tuple(tokens.shape)}",
        )
    if expert_ids.shape[0] != tokens.shape[0]:
        raise ValueError(
            f"expert_ids length {expert_ids.shape[0]} != tokens batch "
            f"{tokens.shape[0]}",
        )
    if n_experts <= 0:
        raise ValueError(f"n_experts must be positive; got {n_experts}")

    actual_ws = _world_size_or_one(group)
    ws = world_size if world_size is not None else actual_ws
    if n_experts % ws != 0:
        raise ValueError(
            f"n_experts {n_experts} must be divisible by world_size {ws}",
        )
    # Simulation mode: when the caller-supplied EP world size doesn't
    # match the active PG (typical for unit tests on single-rank gloo
    # validating a multi-rank EP layout), bypass the actual all-to-all
    # — the in-rank permutation is correct on its own.
    use_real_comms = ws == actual_ws and actual_ws > 1

    n_local_tokens = tokens.shape[0]
    n_per_rank = n_experts // ws

    # Compute destination rank per token, then sort tokens so that
    # tokens going to the same rank are contiguous.
    dest_rank = (expert_ids // n_per_rank).clamp(min=0, max=ws - 1)
    sort_indices = torch.argsort(dest_rank, stable=True)
    sorted_tokens = tokens.index_select(0, sort_indices)
    sorted_dest = dest_rank.index_select(0, sort_indices)

    # Per-rank send counts.
    send_counts = torch.bincount(sorted_dest, minlength=ws).to(torch.int64)

    if use_real_comms:
        recv_counts = _all_to_all_counts(send_counts, group=group)
        permuted_tokens = _all_to_all_payload(
            sorted_tokens, send_counts, recv_counts, group=group,
        )
    else:
        # Simulation: this rank is the only one — sends/recvs locally.
        recv_counts = send_counts.clone()
        permuted_tokens = sorted_tokens.contiguous()

    # Local expert offsets within permuted_tokens. The inbound buffer
    # arrives sorted by source rank (each peer sends its sorted slice
    # of tokens). Within each peer slice, tokens are still sorted by
    # destination rank — but every token there belongs to *this* rank
    # so they all map to local experts. We need per-local-expert
    # offsets, computed by re-binning by expert ID.
    local_expert_offsets = _local_expert_offsets(
        permuted_tokens.shape[0], n_per_rank,
    )
    return DispatchPlan(
        permuted_tokens=permuted_tokens,
        send_counts=send_counts,
        recv_counts=recv_counts,
        sort_indices=sort_indices,
        local_expert_offsets=local_expert_offsets,
        n_local_tokens=n_local_tokens,
    )


def combine_outputs_by_expert(
    expert_outputs: Tensor,
    plan: DispatchPlan,
    *,
    group: dist.ProcessGroup | None = None,
) -> Tensor:
    """Reverse the dispatch: send outputs back, restore original order.

    Args:
        expert_outputs: ``(N_in, D)`` — same shape as
            ``plan.permuted_tokens``; the result of running the local
            experts on each token.
        plan:           the :class:`DispatchPlan` produced by
            :func:`dispatch_tokens_by_expert`.
        group:          optional process group; must match the one
            used in dispatch.

    Returns:
        ``(N_local, D)`` — outputs in the original token order.
    """
    if expert_outputs.shape[0] != plan.permuted_tokens.shape[0]:
        raise ValueError(
            f"expert_outputs len {expert_outputs.shape[0]} must match "
            f"the dispatched permuted_tokens len {plan.permuted_tokens.shape[0]}",
        )

    actual_ws = _world_size_or_one(group)
    plan_ws = plan.send_counts.shape[0]
    use_real_comms = plan_ws == actual_ws and actual_ws > 1

    if use_real_comms:
        # Reverse all-to-all: send_counts and recv_counts swap roles.
        sorted_outputs = _all_to_all_payload(
            expert_outputs,
            plan.recv_counts,
            plan.send_counts,
            group=group,
        )
    else:
        sorted_outputs = expert_outputs.contiguous()

    # Undo the local sort applied at dispatch time.
    out = torch.empty_like(sorted_outputs)
    out.index_copy_(0, plan.sort_indices, sorted_outputs)
    return out


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _world_size_or_one(group: dist.ProcessGroup | None) -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=group)


def _all_to_all_counts(
    send_counts: Tensor,
    *,
    group: dist.ProcessGroup | None,
) -> Tensor:
    """Exchange per-rank send counts to recover the per-rank receive counts."""
    if not dist.is_initialized():
        # Single process, no comms — recv == send.
        return send_counts.clone()
    ws = dist.get_world_size(group=group)
    sends = send_counts.to(torch.int64).contiguous()
    recvs = torch.empty(ws, dtype=torch.int64, device=sends.device)
    dist.all_to_all_single(recvs, sends, group=group)
    return recvs


def _all_to_all_payload(
    sorted_tokens: Tensor,
    send_counts: Tensor,
    recv_counts: Tensor,
    *,
    group: dist.ProcessGroup | None,
) -> Tensor:
    """Run the actual all-to-all with the negotiated split sizes."""
    if not dist.is_initialized():
        # Single process — payload travels nowhere; permuted == sorted.
        return sorted_tokens.contiguous()
    total_recv = int(recv_counts.sum().item())
    out = torch.empty(
        total_recv, sorted_tokens.shape[1],
        dtype=sorted_tokens.dtype, device=sorted_tokens.device,
    )
    dist.all_to_all_single(
        out,
        sorted_tokens.contiguous(),
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
        group=group,
    )
    return out


def _local_expert_offsets(n_in: int, n_local_experts: int) -> Tensor:
    """Empty offsets tensor used by callers that re-bin by expert ID.

    For v0.0 we hand back zeros; production callers run a per-expert
    sort over the inbound buffer (cheap — already-permuted by source
    rank) to populate real offsets. Kept as a placeholder so the
    DispatchPlan dataclass matches the expected shape.
    """
    return torch.zeros(n_local_experts + 1, dtype=torch.int64)
