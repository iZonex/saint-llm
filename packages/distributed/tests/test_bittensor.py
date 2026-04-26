"""Tests for the Bittensor SN miner/validator interface."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from saint_llm_agents.mcp.channel import (
    InMemoryJsonRpcChannel,
    pair_channels,
)
from saint_llm_distributed.bittensor import (
    BittensorMinerServer,
    BittensorValidatorClient,
    MinerInfo,
    MinerResponse,
    WeightedScore,
    aggregate_cohort_scores,
    cohort_consensus_score,
)


class _EchoMiner:
    def __init__(self, prefix: str = "echo") -> None:
        self._prefix = prefix

    def serve(self, query: str) -> str:
        return f"{self._prefix}:{query}"


class _BrokenMiner:
    def serve(self, query: str) -> str:
        del query
        raise RuntimeError("oops")


class _LengthScorer:
    def score(self, query: str, response: str) -> float:
        del query
        return min(1.0, len(response) / 100.0)


def _spawn_miner(
    handler: Any, info: MinerInfo,
) -> tuple[InMemoryJsonRpcChannel, threading.Thread, BittensorMinerServer]:
    client_ch, server_ch = pair_channels()
    server = BittensorMinerServer(handler, info)
    thread = threading.Thread(target=server.serve, args=(server_ch,), daemon=True)
    thread.start()
    return client_ch, thread, server


def _stop(channel: InMemoryJsonRpcChannel, thread: threading.Thread) -> None:
    channel.close()
    thread.join(timeout=2.0)


# ---- miner server ---------------------------------------------------


def test_miner_serve_returns_handler_response() -> None:
    info = MinerInfo(uid="m1", name="echo")
    ch, thread, _ = _spawn_miner(_EchoMiner(), info)
    try:
        ch.send({
            "jsonrpc": "2.0", "id": 1, "method": "miner/serve",
            "params": {"query": "hello"},
        })
        reply = ch.receive()
        assert reply is not None
        assert reply["result"]["response"] == "echo:hello"
        assert reply["result"]["uid"] == "m1"
    finally:
        _stop(ch, thread)


def test_miner_info_returns_metadata() -> None:
    info = MinerInfo(uid="m7", name="alpha", description="test miner")
    ch, thread, _ = _spawn_miner(_EchoMiner(), info)
    try:
        ch.send({"jsonrpc": "2.0", "id": 1, "method": "miner/info"})
        reply = ch.receive()
        assert reply is not None
        result = reply["result"]
        assert result["uid"] == "m7"
        assert result["name"] == "alpha"
        assert result["description"] == "test miner"
    finally:
        _stop(ch, thread)


def test_miner_ping_works() -> None:
    ch, thread, _ = _spawn_miner(_EchoMiner(), MinerInfo(uid="m"))
    try:
        ch.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        reply = ch.receive()
        assert reply is not None
        assert reply["result"] == {}
    finally:
        _stop(ch, thread)


def test_miner_unknown_method_returns_error() -> None:
    ch, thread, _ = _spawn_miner(_EchoMiner(), MinerInfo(uid="m"))
    try:
        ch.send({"jsonrpc": "2.0", "id": 1, "method": "bogus"})
        reply = ch.receive()
        assert reply is not None
        assert reply["error"]["code"] == -32601
    finally:
        _stop(ch, thread)


def test_miner_handler_exception_surfaces_as_error() -> None:
    ch, thread, _ = _spawn_miner(_BrokenMiner(), MinerInfo(uid="m"))
    try:
        ch.send({
            "jsonrpc": "2.0", "id": 1, "method": "miner/serve",
            "params": {"query": "hi"},
        })
        reply = ch.receive()
        assert reply is not None
        assert "error" in reply
        assert "RuntimeError" in reply["error"]["message"]
    finally:
        _stop(ch, thread)


def test_miner_invalid_query_param_rejects() -> None:
    ch, thread, _ = _spawn_miner(_EchoMiner(), MinerInfo(uid="m"))
    try:
        ch.send({
            "jsonrpc": "2.0", "id": 1, "method": "miner/serve",
            "params": {"query": 12345},  # not a string
        })
        reply = ch.receive()
        assert reply is not None
        assert "error" in reply
    finally:
        _stop(ch, thread)


# ---- validator client -----------------------------------------------


def test_validator_query_all_collects_responses_from_each_miner() -> None:
    ch_a, thread_a, _ = _spawn_miner(_EchoMiner("a"), MinerInfo(uid="a"))
    ch_b, thread_b, _ = _spawn_miner(_EchoMiner("b"), MinerInfo(uid="b"))
    try:
        validator = BittensorValidatorClient(
            scorer=_LengthScorer(),
            channels={"a": ch_a, "b": ch_b},
        )
        responses = validator.query_all("hello")
        assert len(responses) == 2
        by_uid = {r.uid: r for r in responses}
        assert by_uid["a"].response == "a:hello"
        assert by_uid["b"].response == "b:hello"
        assert all(r.error is None for r in responses)
    finally:
        _stop(ch_a, thread_a)
        _stop(ch_b, thread_b)


def test_validator_score_responses_sums_to_one_when_normalized() -> None:
    responses = [
        MinerResponse(uid="a", query="x", response="short"),
        MinerResponse(uid="b", query="x", response="a much longer response"),
    ]
    validator = BittensorValidatorClient(scorer=_LengthScorer(), channels={})
    weighted = validator.score_responses(responses, normalize=True)
    assert sum(w.weight for w in weighted) == pytest.approx(1.0)
    # Longer response gets the bigger weight.
    by_uid = {w.uid: w for w in weighted}
    assert by_uid["b"].weight > by_uid["a"].weight


def test_validator_score_responses_unnormalized_returns_raw_scores() -> None:
    responses = [
        MinerResponse(uid="a", query="x", response="short"),
        MinerResponse(uid="b", query="x", response="a much longer response"),
    ]
    validator = BittensorValidatorClient(scorer=_LengthScorer(), channels={})
    weighted = validator.score_responses(responses, normalize=False)
    for w in weighted:
        assert w.weight == w.raw_score


def test_validator_zero_scores_to_errored_responses() -> None:
    responses = [
        MinerResponse(uid="a", query="x", response="ok"),
        MinerResponse(uid="b", query="x", response="", error="failed"),
    ]
    validator = BittensorValidatorClient(scorer=_LengthScorer(), channels={})
    weighted = validator.score_responses(responses, normalize=False)
    by_uid = {w.uid: w for w in weighted}
    assert by_uid["b"].raw_score == 0.0


def test_validator_query_and_score_end_to_end() -> None:
    ch_a, thread_a, _ = _spawn_miner(_EchoMiner("alpha"), MinerInfo(uid="a"))
    ch_b, thread_b, _ = _spawn_miner(
        _EchoMiner("longerprefix"), MinerInfo(uid="b"),
    )
    try:
        validator = BittensorValidatorClient(
            scorer=_LengthScorer(), channels={"a": ch_a, "b": ch_b},
        )
        scores = validator.query_and_score("hi")
        assert isinstance(scores, list)
        assert all(isinstance(s, WeightedScore) for s in scores)
        # Length-based scorer favors the miner with the longer prefix.
        by_uid = {s.uid: s for s in scores}
        assert by_uid["b"].weight > by_uid["a"].weight
    finally:
        _stop(ch_a, thread_a)
        _stop(ch_b, thread_b)


def test_validator_handles_uniform_zero_scores_via_uniform_fallback() -> None:
    """When everyone scored zero, normalize falls back to uniform 1/N."""
    responses = [
        MinerResponse(uid="a", query="x", response=""),
        MinerResponse(uid="b", query="x", response=""),
    ]
    validator = BittensorValidatorClient(scorer=_LengthScorer(), channels={})
    weighted = validator.score_responses(responses, normalize=True)
    assert all(w.weight == 0.5 for w in weighted)


# ---- cohort consensus score -----------------------------------------


def test_consensus_full_overlap_returns_one() -> None:
    score = cohort_consensus_score(
        "q", "the cat sat",
        peer_responses=["the cat sat", "a cat sat"],
    )
    assert score == 1.0


def test_consensus_no_overlap_returns_zero() -> None:
    score = cohort_consensus_score(
        "q", "alpha beta",
        peer_responses=["gamma delta", "epsilon zeta"],
    )
    assert score == 0.0


def test_consensus_partial_overlap() -> None:
    score = cohort_consensus_score(
        "q", "cat dog",
        peer_responses=["cat fish", "fish bird", "dog bird"],
    )
    # First peer shares 'cat'; third peer shares 'dog'; second peer overlaps neither.
    assert score == pytest.approx(2 / 3, abs=1e-6)


def test_consensus_empty_response_returns_zero() -> None:
    assert cohort_consensus_score("q", "", peer_responses=["x"]) == 0.0


def test_consensus_no_peers_returns_zero() -> None:
    assert cohort_consensus_score("q", "x", peer_responses=[]) == 0.0


# ---- aggregate cohort scores ----------------------------------------


def test_aggregate_mean() -> None:
    assert aggregate_cohort_scores([0.1, 0.3, 0.5], mode="mean") == pytest.approx(0.3)


def test_aggregate_median() -> None:
    assert aggregate_cohort_scores([0.1, 0.5, 0.9], mode="median") == 0.5


def test_aggregate_max_and_min() -> None:
    assert aggregate_cohort_scores([0.1, 0.5, 0.9], mode="max") == 0.9
    assert aggregate_cohort_scores([0.1, 0.5, 0.9], mode="min") == 0.1


def test_aggregate_empty_returns_zero() -> None:
    assert aggregate_cohort_scores([], mode="mean") == 0.0


def test_aggregate_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        aggregate_cohort_scores([0.1], mode="bogus")
