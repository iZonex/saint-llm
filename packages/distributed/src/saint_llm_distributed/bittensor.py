"""Bittensor SN miner / validator interface — Web3-style decentralized inference.

Bittensor subnets pair *miners* (run inference, return predictions)
with *validators* (issue queries, score responses, set weights for
the chain to mint TAO against). Saint-llm is designed to plug in as
either role without depending on the heavy ``bittensor`` SDK — this
module ships the protocol shapes and a JSON-RPC reference adapter
that production deploys can swap for a real Bittensor wallet/axon
binding.

Design:

* :class:`MinerHandler` Protocol — receives a query string, returns a
  response string. Stateful (the miner instance owns the model).
* :class:`ValidatorScorer` Protocol — given ``(query, response)``,
  emits a numeric score in ``[0, 1]``. Stateful so a real validator
  can hold its scoring model.
* :class:`BittensorMinerServer` — wraps a :class:`MinerHandler` and
  serves queries over our existing :class:`JsonRpcChannel`. Plugs
  into the same in-memory pair / stdio transports we use for MCP +
  A2A.
* :class:`BittensorValidatorClient` — opposite end: dispatches a
  query, collects the response, scores it, returns a dict suitable
  for emitting on-chain weights.

This is the v0.0 *interface*. A real Bittensor SN integration:

* uses ``bittensor.axon`` instead of our JsonRpcChannel for transport
  (axon is a Substrate-signed gRPC service).
* persists weights to chain via ``bittensor.subtensor.set_weights``.
* queries miners' UIDs via ``bittensor.metagraph``.

Production swaps the transport layer; the :class:`MinerHandler` /
:class:`ValidatorScorer` Protocols stay identical.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Try to reuse the JsonRpcChannel from the agents package without
# making it a hard dependency. Fall back to a duck-typed Protocol.
try:
    from saint_llm_agents.mcp.channel import JsonRpcChannel
except ImportError:  # pragma: no cover — used only when agents isn't installed
    JsonRpcChannel = Any  # type: ignore[assignment, misc]


class MinerHandler(Protocol):
    """A miner answers queries from the network."""

    def serve(self, query: str) -> str: ...


class ValidatorScorer(Protocol):
    """A validator scores miner responses."""

    def score(self, query: str, response: str) -> float: ...


@dataclass(frozen=True)
class MinerInfo:
    """Miner metadata advertised over the protocol."""

    uid: str
    name: str = ""
    description: str = ""
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MinerResponse:
    """One miner's response to a validator query."""

    uid: str
    query: str
    response: str
    error: str | None = None


@dataclass(frozen=True)
class WeightedScore:
    """Validator's per-miner scoring outcome."""

    uid: str
    raw_score: float
    weight: float
    response: str


class BittensorMinerServer:
    """JSON-RPC server exposing a :class:`MinerHandler` to the network.

    Methods served:
        ``ping``           — round-trip check
        ``miner/info``     — return :class:`MinerInfo`
        ``miner/serve``    — call ``handler.serve(query)``, return response
    """

    def __init__(
        self,
        handler: MinerHandler,
        info: MinerInfo,
    ) -> None:
        self._handler = handler
        self._info = info

    def handle_one(self, msg: Mapping[str, Any], channel: Any) -> None:
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        if method is None:
            return
        is_notification = msg_id is None
        try:
            if method == "ping":
                result: Any = {}
            elif method == "miner/info":
                result = {
                    "uid": self._info.uid,
                    "name": self._info.name,
                    "description": self._info.description,
                    "metadata": dict(self._info.metadata or {}),
                }
            elif method == "miner/serve":
                query = params.get("query")
                if not isinstance(query, str):
                    raise ValueError("miner/serve requires 'query: str'")
                response = self._handler.serve(query)
                result = {"uid": self._info.uid, "response": str(response)}
            else:
                if not is_notification:
                    channel.send({
                        "jsonrpc": "2.0", "id": msg_id,
                        "error": {"code": -32601, "message": f"method not found: {method}"},
                    })
                return
        except Exception as exc:
            logger.exception("bittensor miner: %s crashed", method)
            if not is_notification:
                channel.send({
                    "jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"},
                })
            return
        if not is_notification:
            channel.send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def serve(self, channel: Any) -> None:
        while True:
            msg = channel.receive()
            if msg is None:
                return
            self.handle_one(msg, channel)


class BittensorValidatorClient:
    """Validator-side: dispatch queries, collect responses, score them.

    Args:
        scorer:   the per-miner scorer.
        channels: ``{uid: channel}`` map for the miners to query.
            Channels must satisfy the :class:`JsonRpcChannel` shape
            (``send`` + ``receive``).
    """

    def __init__(
        self,
        *,
        scorer: ValidatorScorer,
        channels: Mapping[str, Any],
    ) -> None:
        self._scorer = scorer
        self._channels = dict(channels)
        self._next_id = 1

    def query_all(self, query: str) -> list[MinerResponse]:
        """Send the query to every connected miner; collect responses."""
        out: list[MinerResponse] = []
        for uid, channel in self._channels.items():
            msg_id = self._next_id
            self._next_id += 1
            channel.send({
                "jsonrpc": "2.0", "id": msg_id, "method": "miner/serve",
                "params": {"query": query},
            })
            reply = self._read_until(channel, msg_id)
            if reply is None:
                out.append(MinerResponse(
                    uid=uid, query=query, response="",
                    error="channel closed before reply",
                ))
                continue
            if "error" in reply:
                err = reply["error"]
                out.append(MinerResponse(
                    uid=uid, query=query, response="",
                    error=f"{err.get('code')}: {err.get('message')}",
                ))
                continue
            result = reply.get("result") or {}
            out.append(MinerResponse(
                uid=uid, query=query,
                response=str(result.get("response", "")),
                error=None,
            ))
        return out

    def score_responses(
        self,
        responses: Sequence[MinerResponse],
        *,
        normalize: bool = True,
    ) -> list[WeightedScore]:
        """Apply the scorer to each response; optionally normalize to weights summing to 1."""
        raw: list[tuple[str, float, str]] = []
        for r in responses:
            if r.error is not None or not r.response:
                raw.append((r.uid, 0.0, r.response))
                continue
            score = float(self._scorer.score(r.query, r.response))
            raw.append((r.uid, score, r.response))

        total = sum(s for _, s, _ in raw)
        out: list[WeightedScore] = []
        for uid, raw_score, resp in raw:
            if normalize and total > 0.0:
                weight = raw_score / total
            elif normalize:
                weight = 1.0 / max(len(raw), 1)
            else:
                weight = raw_score
            out.append(WeightedScore(
                uid=uid, raw_score=raw_score, weight=weight, response=resp,
            ))
        return out

    def query_and_score(
        self, query: str, *, normalize: bool = True,
    ) -> list[WeightedScore]:
        """Convenience: query all + score in one call."""
        return self.score_responses(self.query_all(query), normalize=normalize)

    @staticmethod
    def _read_until(channel: Any, msg_id: int) -> Mapping[str, Any] | None:
        while True:
            reply = channel.receive()
            if reply is None:
                return None
            if reply.get("id") == msg_id:
                return reply


def cohort_consensus_score(
    query: str, response: str, *, peer_responses: Sequence[str],
) -> float:
    """Score by similarity to the consensus of peer responses.

    Cheap proxy for "this miner agrees with the cohort." Use as a
    placeholder :class:`ValidatorScorer` in tests + early validators
    that don't yet have a real reward model. Returns the fraction of
    peer responses that share at least one whitespace-token with this
    response.
    """
    del query
    if not response or not peer_responses:
        return 0.0
    pred_tokens = set(response.lower().split())
    if not pred_tokens:
        return 0.0
    n_overlap = sum(
        1 for peer in peer_responses
        if pred_tokens & set(peer.lower().split())
    )
    return n_overlap / len(peer_responses)


def aggregate_cohort_scores(
    raw_scores: Sequence[float], *, mode: str = "mean",
) -> float:
    """Aggregate per-validator scores into one cluster-wide score.

    Modes: ``"mean"``, ``"median"``, ``"max"``, ``"min"``. Used when
    multiple validators score the same miner and the chain needs one
    final number to mint against.
    """
    if not raw_scores:
        return 0.0
    if mode == "mean":
        return float(sum(raw_scores) / len(raw_scores))
    if mode == "median":
        return float(statistics.median(raw_scores))
    if mode == "max":
        return float(max(raw_scores))
    if mode == "min":
        return float(min(raw_scores))
    raise ValueError(f"unknown mode {mode!r}")
