"""Multi-agent Runtime — round-robin orchestrator with per-agent inboxes.

This is the v0.0 cut: simplest thing that makes "multi-agent" actually
mean more than ad-hoc composition.

Model:

* Two or more :class:`Agent` instances registered by name.
* One inbox per agent (a list of :class:`Message`).
* :meth:`Runtime.send` posts a user message into a specific inbox; this
  is how the human (or upstream system) hands work to the agent network.
* :meth:`Runtime.tick` advances one agent: drains its inbox, runs
  :meth:`Agent.step`, appends produced messages to the global
  ``transcript``, and forwards the agent's assistant output to the
  *next* agent's inbox — converted from ``role="assistant"`` to
  ``role="user"`` so the receiver doesn't think the previous turn was
  its own. Tool result messages stay in the producing agent's history;
  they are not forwarded.
* :meth:`Runtime.run` ticks ``max_ticks`` times and returns the full
  transcript.

Limitations (deliberately deferred to v0.1):

* No explicit recipient on messages — assistant outputs always go to
  ``cursor+1``. For arbitrary routing patterns we'd need ``Message.to``
  plus a routing policy.
* No termination heuristic — runs until ``max_ticks`` is hit. A "stop"
  signal (e.g. a sentinel message or a policy-side flag) is the
  natural v0.1 addition.
* No parallel scheduling. One agent ticks at a time.
"""

from __future__ import annotations

from collections.abc import Sequence

from saint_llm_agents.agent import Agent
from saint_llm_agents.message import Message


class Runtime:
    """Round-robin orchestrator for a fixed set of agents.

    Args:
        agents: at least two :class:`Agent` instances. Names must be
            unique within the runtime.

    Raises:
        ValueError: if fewer than two agents are supplied or names
            collide.
    """

    def __init__(self, agents: Sequence[Agent]) -> None:
        if len(agents) < 2:
            raise ValueError(f"Runtime needs at least 2 agents; got {len(agents)}")
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise ValueError(f"agent names must be unique; got {names}")

        self._agents: list[Agent] = list(agents)
        self._inboxes: dict[str, list[Message]] = {a.name: [] for a in agents}
        self._cursor: int = 0
        self._transcript: list[Message] = []

    @property
    def agents(self) -> tuple[Agent, ...]:
        return tuple(self._agents)

    @property
    def transcript(self) -> list[Message]:
        return list(self._transcript)

    @property
    def cursor(self) -> int:
        return self._cursor

    def inbox_for(self, name: str) -> list[Message]:
        if name not in self._inboxes:
            raise KeyError(f"unknown agent {name!r}")
        return list(self._inboxes[name])

    def send(self, *, to: str, content: str, sender: str = "user") -> None:
        """Post a user-role message into agent ``to``'s inbox.

        ``sender`` is the ``Message.name`` stamped on the message; defaults
        to ``"user"``. The message is also recorded in the transcript so
        external observers see the prompt that started the dialogue.
        """
        if to not in self._inboxes:
            raise KeyError(f"unknown agent {to!r}")
        msg = Message(role="user", content=content, name=sender)
        self._inboxes[to].append(msg)
        self._transcript.append(msg)

    def tick(self) -> list[Message]:
        """Run one agent for one step; route its assistant output forward.

        Returns the messages produced by the stepped agent (assistant
        plus any tool results).
        """
        agent = self._agents[self._cursor]
        inbox = self._inboxes[agent.name]
        self._inboxes[agent.name] = []

        produced = agent.step(inbox)
        self._transcript.extend(produced)

        next_agent = self._agents[(self._cursor + 1) % len(self._agents)]
        for m in produced:
            if m.role == "assistant" and m.content:
                # Cross-agent boundary: assistant → user re-stamp so the
                # receiver does not mistake the prior turn for its own.
                self._inboxes[next_agent.name].append(
                    Message(role="user", content=m.content, name=agent.name),
                )

        self._cursor = (self._cursor + 1) % len(self._agents)
        return produced

    def run(self, *, max_ticks: int = 20) -> list[Message]:
        """Tick up to ``max_ticks`` times and return the transcript."""
        if max_ticks <= 0:
            raise ValueError(f"max_ticks must be positive; got {max_ticks}")
        for _ in range(max_ticks):
            self.tick()
        return self.transcript
