"""Tests for expert-parallel dispatch / combine on single-rank gloo."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from saint_llm_distributed.expert_parallel import (
    DispatchPlan,
    combine_outputs_by_expert,
    dispatch_tokens_by_expert,
)
from saint_llm_distributed.fsdp import ensure_local_gloo_pg


@pytest.fixture(scope="session", autouse=True)
def _gloo_pg() -> None:
    ensure_local_gloo_pg()
    yield


def test_dispatch_returns_plan_with_expected_fields() -> None:
    tokens = torch.randn(8, 4)
    expert_ids = torch.randint(0, 4, (8,))
    plan = dispatch_tokens_by_expert(
        tokens, expert_ids, n_experts=4, world_size=1,
    )
    assert isinstance(plan, DispatchPlan)
    assert plan.permuted_tokens.shape == (8, 4)
    assert plan.send_counts.shape == (1,)
    assert plan.recv_counts.shape == (1,)
    assert plan.sort_indices.shape == (8,)
    assert plan.n_local_tokens == 8


def test_dispatch_preserves_token_count_single_rank() -> None:
    """All tokens stay on the single rank; permuted shape == input shape."""
    tokens = torch.randn(16, 8)
    expert_ids = torch.randint(0, 4, (16,))
    plan = dispatch_tokens_by_expert(
        tokens, expert_ids, n_experts=4, world_size=1,
    )
    assert plan.permuted_tokens.shape[0] == 16
    assert int(plan.send_counts.sum().item()) == 16
    assert int(plan.recv_counts.sum().item()) == 16


def test_dispatch_sorts_tokens_by_destination_rank() -> None:
    """sort_indices reorders tokens so same-destination tokens are contiguous."""
    tokens = torch.arange(8, dtype=torch.float32).unsqueeze(-1)  # (8, 1)
    # 4 experts, world_size=2 -> rank 0 owns experts {0,1}, rank 1 owns {2,3}.
    # Tokens [0,1,2,3,4,5,6,7] map to experts [3,1,2,0,3,1,0,2].
    expert_ids = torch.tensor([3, 1, 2, 0, 3, 1, 0, 2])
    plan = dispatch_tokens_by_expert(
        tokens, expert_ids, n_experts=4, world_size=2,
    )
    # Send counts: rank 0 (experts 0/1) gets tokens [3,1,5,6] -> 4 tokens.
    # Rank 1 (experts 2/3) gets tokens [0,2,4,7] -> 4 tokens.
    assert plan.send_counts.tolist() == [4, 4]


def test_combine_recovers_original_order() -> None:
    """Round-trip: dispatch then combine returns tokens to their original positions."""
    tokens = torch.arange(12, dtype=torch.float32).reshape(12, 1)
    expert_ids = torch.tensor([3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1])
    plan = dispatch_tokens_by_expert(
        tokens, expert_ids, n_experts=4, world_size=1,
    )
    # Identity expert: pass through.
    expert_outputs = plan.permuted_tokens.clone()
    restored = combine_outputs_by_expert(expert_outputs, plan)
    assert torch.allclose(restored, tokens)


def test_combine_with_perturbed_outputs_recovers_per_token_signal() -> None:
    """An expert that adds 100 to its tokens; combine must put offsets in original order."""
    tokens = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    expert_ids = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    plan = dispatch_tokens_by_expert(
        tokens, expert_ids, n_experts=2, world_size=1,
    )
    out = plan.permuted_tokens + 100.0
    restored = combine_outputs_by_expert(out, plan)
    expected = tokens + 100.0
    assert torch.allclose(restored, expected)


def test_dispatch_rejects_n_experts_not_divisible_by_world_size() -> None:
    tokens = torch.randn(4, 2)
    expert_ids = torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError, match="divisible by world_size"):
        dispatch_tokens_by_expert(
            tokens, expert_ids, n_experts=5, world_size=2,
        )


def test_dispatch_rejects_non_2d_tokens() -> None:
    tokens = torch.randn(4)
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        dispatch_tokens_by_expert(
            tokens, torch.zeros(4, dtype=torch.long), n_experts=4, world_size=1,
        )


def test_dispatch_rejects_mismatched_expert_ids_length() -> None:
    tokens = torch.randn(4, 2)
    expert_ids = torch.zeros(5, dtype=torch.long)
    with pytest.raises(ValueError, match="length"):
        dispatch_tokens_by_expert(
            tokens, expert_ids, n_experts=4, world_size=1,
        )


def test_dispatch_zero_n_experts_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        dispatch_tokens_by_expert(
            torch.randn(4, 2), torch.zeros(4, dtype=torch.long),
            n_experts=0, world_size=1,
        )


def test_combine_rejects_size_mismatch() -> None:
    tokens = torch.randn(4, 2)
    expert_ids = torch.zeros(4, dtype=torch.long)
    plan = dispatch_tokens_by_expert(
        tokens, expert_ids, n_experts=2, world_size=1,
    )
    bad_outputs = torch.randn(7, 2)  # wrong batch dim
    with pytest.raises(ValueError, match="must match"):
        combine_outputs_by_expert(bad_outputs, plan)


def test_dispatch_uses_world_pg_when_initialized() -> None:
    """When PG is up, world_size defaults to dist.get_world_size()."""
    tokens = torch.randn(4, 2)
    expert_ids = torch.tensor([0, 1, 0, 1])
    plan = dispatch_tokens_by_expert(tokens, expert_ids, n_experts=2)
    # Single-rank gloo -> world_size=1.
    assert dist.get_world_size() == 1
    assert plan.send_counts.shape == (1,)


def test_dispatch_handles_all_tokens_to_same_expert() -> None:
    tokens = torch.randn(4, 3)
    expert_ids = torch.full((4,), 1, dtype=torch.long)
    plan = dispatch_tokens_by_expert(
        tokens, expert_ids, n_experts=4, world_size=1,
    )
    # Every token goes to the single rank; sort puts them in expert-1 order.
    assert plan.permuted_tokens.shape == (4, 3)
    restored = combine_outputs_by_expert(plan.permuted_tokens, plan)
    assert torch.allclose(restored, tokens)


def test_dispatch_round_trip_with_random_routing() -> None:
    """Random expert assignments still round-trip cleanly."""
    torch.manual_seed(0)
    tokens = torch.randn(32, 8)
    expert_ids = torch.randint(0, 4, (32,))
    plan = dispatch_tokens_by_expert(
        tokens, expert_ids, n_experts=4, world_size=1,
    )
    restored = combine_outputs_by_expert(plan.permuted_tokens, plan)
    assert torch.allclose(restored, tokens)


def test_dispatch_preserves_dtype_and_device() -> None:
    tokens = torch.randn(4, 2, dtype=torch.float32)
    expert_ids = torch.zeros(4, dtype=torch.long)
    plan = dispatch_tokens_by_expert(
        tokens, expert_ids, n_experts=2, world_size=1,
    )
    assert plan.permuted_tokens.dtype == torch.float32
    assert plan.permuted_tokens.device == tokens.device


def test_local_expert_offsets_shape_in_plan() -> None:
    plan = dispatch_tokens_by_expert(
        torch.randn(4, 2), torch.zeros(4, dtype=torch.long),
        n_experts=4, world_size=2,
    )
    # n_per_rank = 2 -> offsets length 3 (one extra for the sentinel end).
    assert plan.local_expert_offsets.shape == (3,)
