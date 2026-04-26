"""Tests for hybrid ZeRO bucket assignment."""

from __future__ import annotations

import pytest
import torch
from saint_llm_distributed.zero import (
    BucketAssignment,
    adamw_param_cost,
    hybrid_zero_assign,
    muon_param_cost,
    pack_buckets_by_cost,
)


def test_pack_buckets_simple_two_ranks() -> None:
    """Items 4, 3, 2, 1 across 2 buckets -> [[0, 3], [1, 2]] (LPT)."""
    a = pack_buckets_by_cost([4.0, 3.0, 2.0, 1.0], n_buckets=2)
    # Sorted desc: [0(4), 1(3), 2(2), 3(1)].
    # Place 0 -> bucket 0 (cost 4). Place 1 -> bucket 1 (cost 3).
    # Place 2 -> bucket 1 (cost 5). Place 3 -> bucket 0 (cost 5).
    assert sum(a.per_rank_cost) == 10.0
    assert a.per_rank_cost[0] == a.per_rank_cost[1] == 5.0


def test_pack_buckets_empty_input() -> None:
    a = pack_buckets_by_cost([], n_buckets=4)
    assert a.n_ranks == 4
    assert all(len(b) == 0 for b in a.buckets)
    assert all(c == 0.0 for c in a.per_rank_cost)


def test_pack_buckets_single_rank_holds_everything() -> None:
    a = pack_buckets_by_cost([3.0, 1.0, 4.0, 1.0, 5.0], n_buckets=1)
    assert len(a.buckets[0]) == 5
    assert a.per_rank_cost[0] == 14.0


def test_pack_buckets_more_ranks_than_items() -> None:
    a = pack_buckets_by_cost([1.0, 2.0], n_buckets=5)
    # 2 items, 5 ranks -> 3 ranks empty. Heaviest item goes to lightest.
    nonempty = [i for i, b in enumerate(a.buckets) if b]
    assert len(nonempty) == 2


def test_pack_buckets_imbalance_metric() -> None:
    a = pack_buckets_by_cost([10.0, 1.0, 1.0, 1.0], n_buckets=2)
    # Rank 0 gets 10; rank 1 gets 1+1+1=3. Imbalance = (10-3)/10 = 0.7.
    assert a.imbalance == pytest.approx(0.7, abs=1e-6)


def test_pack_buckets_balanced_returns_zero_imbalance() -> None:
    a = pack_buckets_by_cost([4.0, 4.0, 4.0, 4.0], n_buckets=4)
    assert a.imbalance == 0.0


def test_pack_buckets_rejects_zero_buckets() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        pack_buckets_by_cost([1.0, 2.0], n_buckets=0)


def test_pack_buckets_rejects_negative_costs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        pack_buckets_by_cost([1.0, -2.0], n_buckets=2)


def test_pack_buckets_zero_cost_items_still_assigned() -> None:
    a = pack_buckets_by_cost([0.0, 0.0, 0.0], n_buckets=2)
    total_assigned = sum(len(b) for b in a.buckets)
    assert total_assigned == 3


def test_muon_param_cost_2d_uses_dim_product() -> None:
    t = torch.zeros(128, 256)
    assert muon_param_cost(t) == 128 * 256


def test_muon_param_cost_1d_falls_back_to_numel() -> None:
    t = torch.zeros(64)
    assert muon_param_cost(t) == 64


def test_muon_param_cost_3d_uses_numel() -> None:
    t = torch.zeros(2, 4, 8)
    assert muon_param_cost(t) == 64


def test_adamw_param_cost_is_numel() -> None:
    t = torch.zeros(3, 5)
    assert adamw_param_cost(t) == 15


def test_hybrid_zero_assign_partitions_optimizer_groups() -> None:
    """Muon and AdamW packed independently then concatenated."""
    params = [
        torch.zeros(100, 100),  # 0: muon, cost 10000
        torch.zeros(100),       # 1: adamw, cost 100
        torch.zeros(50, 50),    # 2: muon, cost 2500
        torch.zeros(200),       # 3: adamw, cost 200
    ]
    kinds: list = ["muon", "adamw", "muon", "adamw"]
    a = hybrid_zero_assign(params, kinds, n_buckets=2)
    # All four params are assigned across the 2 buckets.
    total = sum(len(b) for b in a.buckets)
    assert total == 4
    # Total cost = 10000 + 2500 + 100 + 200 = 12800.
    assert sum(a.per_rank_cost) == 12800.0


def test_hybrid_zero_assign_balances_muon_and_adamw_separately() -> None:
    """Each rank gets approximately equal cost from each pool."""
    params = [
        torch.zeros(100, 100),  # muon
        torch.zeros(100, 100),  # muon
        torch.zeros(1000),      # adamw
        torch.zeros(1000),      # adamw
    ]
    kinds: list = ["muon", "muon", "adamw", "adamw"]
    a = hybrid_zero_assign(params, kinds, n_buckets=2)
    # Both muon items have cost 10000, both adamw items have cost 1000.
    # Each rank should get one of each.
    assert a.per_rank_cost[0] == a.per_rank_cost[1] == 11000.0


def test_hybrid_zero_assign_handles_only_muon() -> None:
    params = [torch.zeros(100, 100), torch.zeros(50, 50)]
    a = hybrid_zero_assign(params, ["muon", "muon"], n_buckets=2)
    # Largest goes to rank 0; smaller to rank 1.
    assert a.per_rank_cost[0] == 10000.0
    assert a.per_rank_cost[1] == 2500.0


def test_hybrid_zero_assign_handles_only_adamw() -> None:
    params = [torch.zeros(100), torch.zeros(50)]
    a = hybrid_zero_assign(params, ["adamw", "adamw"], n_buckets=2)
    assert a.per_rank_cost[0] == 100.0
    assert a.per_rank_cost[1] == 50.0


def test_hybrid_zero_assign_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        hybrid_zero_assign(
            [torch.zeros(4, 4)], ["muon", "adamw"], n_buckets=2,
        )


def test_hybrid_zero_assign_rejects_unknown_optimizer_kind() -> None:
    with pytest.raises(ValueError, match="must be 'muon' or 'adamw'"):
        hybrid_zero_assign(
            [torch.zeros(4, 4)], ["lion"], n_buckets=2,  # type: ignore[list-item]
        )


def test_bucket_assignment_imbalance_zero_when_empty() -> None:
    a = BucketAssignment(buckets=((), ()), per_rank_cost=(0.0, 0.0))
    assert a.imbalance == 0.0
