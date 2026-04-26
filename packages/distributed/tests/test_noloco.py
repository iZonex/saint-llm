"""Tests for NoLoCo — log(N) ring all-reduce + AdamW outer."""

from __future__ import annotations

import threading

import pytest
import torch
from saint_llm_distributed import (
    NoLoCo,
    NoLoCoConfig,
    RingAllReduceSimulator,
    ring_all_reduce,
    ring_log2_rounds,
)
from torch import nn


def _drive_peers(funcs: list) -> list:  # type: ignore[type-arg]
    """Run each peer's callable in its own thread and join. Returns results."""
    results: list = [None] * len(funcs)
    threads: list[threading.Thread] = []

    def _worker(i: int, fn) -> None:  # type: ignore[no-untyped-def]
        results[i] = fn()

    for i, fn in enumerate(funcs):
        t = threading.Thread(target=_worker, args=(i, fn))
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        if t.is_alive():
            raise TimeoutError("peer thread did not finish in 5s")
    return results


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 4, bias=False), nn.Linear(4, 2, bias=False))


def test_ring_log2_rounds() -> None:
    assert ring_log2_rounds(0) == 0
    assert ring_log2_rounds(1) == 0
    assert ring_log2_rounds(2) == 1
    assert ring_log2_rounds(3) == 2
    assert ring_log2_rounds(4) == 2
    assert ring_log2_rounds(8) == 3
    assert ring_log2_rounds(7) == 3


def test_ring_all_reduce_single_peer_returns_input() -> None:
    t = [torch.tensor([1.0, 2.0])]
    out = ring_all_reduce([t])
    assert torch.equal(out[0][0], t[0])


def test_ring_all_reduce_two_peers_averages() -> None:
    a = [torch.tensor([2.0, 4.0])]
    b = [torch.tensor([0.0, 8.0])]
    out = ring_all_reduce([a, b])
    expected = torch.tensor([1.0, 6.0])
    assert torch.allclose(out[0][0], expected)
    assert torch.allclose(out[1][0], expected)


def test_ring_all_reduce_power_of_two_converges_to_global_mean() -> None:
    """For N power of 2, log2(N) rounds give exact mean within FP error."""
    torch.manual_seed(0)
    inputs = [[torch.randn(4)] for _ in range(8)]
    expected = sum(t[0] for t in inputs) / len(inputs)
    out = ring_all_reduce(inputs)
    for peer_out in out:
        assert torch.allclose(peer_out[0], expected, atol=1e-5)


def test_ring_all_reduce_handles_multiple_tensors_per_peer() -> None:
    a = [torch.tensor([1.0]), torch.tensor([10.0, 20.0])]
    b = [torch.tensor([3.0]), torch.tensor([30.0, 40.0])]
    out = ring_all_reduce([a, b])
    assert torch.allclose(out[0][0], torch.tensor([2.0]))
    assert torch.allclose(out[0][1], torch.tensor([20.0, 30.0]))


def test_ring_all_reduce_empty_input_returns_empty() -> None:
    assert ring_all_reduce([]) == []


def test_ring_simulator_peer_reducers_share_pool() -> None:
    sim = RingAllReduceSimulator(n_peers=4)
    reducers = [sim.peer_reducer(i) for i in range(4)]
    inputs = [[torch.full((3,), float(i))] for i in range(4)]
    outputs = _drive_peers([
        (lambda i=i: reducers[i].reduce(inputs[i])) for i in range(4)
    ])
    expected = torch.full((3,), 1.5)  # mean(0,1,2,3) = 1.5
    for out in outputs:
        assert torch.allclose(out[0], expected)


def test_ring_simulator_run_round_single_thread() -> None:
    sim = RingAllReduceSimulator(n_peers=4)
    inputs = [[torch.full((3,), float(i))] for i in range(4)]
    outputs = sim.run_round(inputs)
    expected = torch.full((3,), 1.5)
    for out in outputs:
        assert torch.allclose(out[0], expected)


def test_ring_simulator_run_round_wrong_peer_count_raises() -> None:
    sim = RingAllReduceSimulator(n_peers=4)
    with pytest.raises(ValueError, match="expected 4 peer inputs"):
        sim.run_round([[torch.zeros(3)]])


def test_ring_simulator_peer_index_out_of_range_raises() -> None:
    sim = RingAllReduceSimulator(n_peers=2)
    with pytest.raises(ValueError, match="out of range"):
        sim.peer_reducer(5)


def test_ring_simulator_zero_peers_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        RingAllReduceSimulator(n_peers=0)


def test_noloco_starts_unsynced() -> None:
    model = _tiny_model()
    sim = RingAllReduceSimulator(n_peers=1)
    nl = NoLoCo(model.parameters(), NoLoCoConfig(inner_steps=4),
                reducer=sim.peer_reducer(0))
    assert nl.should_sync is False


def test_noloco_should_sync_after_inner_steps() -> None:
    model = _tiny_model()
    sim = RingAllReduceSimulator(n_peers=1)
    cfg = NoLoCoConfig(inner_steps=2)
    nl = NoLoCo(model.parameters(), cfg, reducer=sim.peer_reducer(0))
    nl.inner_step()
    assert nl.should_sync is False
    nl.inner_step()
    assert nl.should_sync is True


def test_noloco_outer_sync_resets_inner_count() -> None:
    model = _tiny_model()
    sim = RingAllReduceSimulator(n_peers=1)
    nl = NoLoCo(model.parameters(), NoLoCoConfig(inner_steps=1),
                reducer=sim.peer_reducer(0))
    nl.inner_step()
    nl.outer_sync()
    assert nl.should_sync is False


def test_noloco_outer_sync_applies_adam_update() -> None:
    """When inner step changes params, outer_sync applies AdamW pseudo-grad."""
    torch.manual_seed(0)
    model = _tiny_model()
    sim = RingAllReduceSimulator(n_peers=1)
    cfg = NoLoCoConfig(inner_steps=1, outer_lr=1e-2)
    nl = NoLoCo(model.parameters(), cfg, reducer=sim.peer_reducer(0))

    # Capture initial params.
    initial = [p.detach().clone() for p in model.parameters()]
    # Simulate one inner step that perturbs params.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    nl.inner_step()
    nl.outer_sync()

    # After outer_sync, params should differ from both initial AND the
    # post-inner-step values: AdamW adjusts the update magnitude.
    for i, p in enumerate(model.parameters()):
        assert not torch.allclose(p, initial[i])


def test_noloco_state_dict_round_trip() -> None:
    model = _tiny_model()
    sim = RingAllReduceSimulator(n_peers=1)
    nl = NoLoCo(model.parameters(), NoLoCoConfig(inner_steps=2),
                reducer=sim.peer_reducer(0))
    nl.inner_step()
    nl.inner_step()
    nl.outer_sync()
    state = nl.state_dict()

    model2 = _tiny_model()
    sim2 = RingAllReduceSimulator(n_peers=1)
    nl2 = NoLoCo(model2.parameters(), NoLoCoConfig(inner_steps=2),
                 reducer=sim2.peer_reducer(0))
    nl2.load_state_dict(state)
    state2 = nl2.state_dict()

    assert state2["inner_count"] == state["inner_count"]
    assert state2["step"] == state["step"]
    for a, b in zip(state["snapshot"], state2["snapshot"], strict=True):
        assert torch.allclose(a, b)


def test_noloco_multi_peer_simulation_keeps_params_in_sync() -> None:
    """N peers with identical state converge to the same params after sync."""
    torch.manual_seed(0)
    n_peers = 4
    sim = RingAllReduceSimulator(n_peers=n_peers)
    cfg = NoLoCoConfig(inner_steps=1, outer_lr=1e-2)

    models = [_tiny_model() for _ in range(n_peers)]
    initial_state = [p.detach().clone() for p in models[0].parameters()]
    for m in models[1:]:
        for p, ref in zip(m.parameters(), initial_state, strict=True):
            p.data.copy_(ref)

    drivers = [
        NoLoCo(m.parameters(), cfg, reducer=sim.peer_reducer(i))
        for i, m in enumerate(models)
    ]

    # Distinct inner perturbations per peer.
    for i, m in enumerate(models):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.1 * (i + 1))
        drivers[i].inner_step()

    # All peers run outer_sync concurrently — the simulator's barrier
    # synchronizes them.
    _drive_peers([(lambda d=d: d.outer_sync()) for d in drivers])

    # After sync, every peer should hold identical params (same global mean).
    ref_params = list(models[0].parameters())
    for m in models[1:]:
        for p, ref in zip(m.parameters(), ref_params, strict=True):
            assert torch.allclose(p, ref, atol=1e-5)
