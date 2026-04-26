"""Agent — one think → act → observe loop with tool execution.

State is just a list of :class:`Message` (the conversation so far). Each
:meth:`Agent.step` cycle:

1. Append any inbound messages to history.
2. Ask the policy what to do next, given ``history`` and the tools
   currently registered.
3. Append the assistant response to history.
4. If the assistant requested any tool calls, execute each in order via
   the registry, append each result message to history, and return them
   together with the assistant message.

:meth:`Agent.run` is the convenience driver: send a user message, then
keep stepping with empty inbox until the policy stops emitting tool
calls (or ``max_steps`` is hit). It returns the full history.

Multi-agent orchestration (an explicit Runtime that routes messages
between agents and ticks them in a deterministic order) is the next
piece — for now you can compose multiple Agents manually by calling
``other.step(producer.step(...))``.
"""

from __future__ import annotations

from collections.abc import Sequence

from saint_llm_agents.message import Message
from saint_llm_agents.policy import Policy
from saint_llm_agents.tool import ToolRegistry


class Agent:
    """Stateful agent: policy + tools + conversation history.

    Args:
        name:   short identifier (also used as :attr:`Message.name` when
            stamping outbound messages, so external observers can tell
            agents apart in a multi-agent transcript).
        policy: the brain. See :mod:`saint_llm_agents.policy`.
        system: optional system-prompt seed. If non-empty, prepended as
            a ``role="system"`` message at construction time.
        tools:  registry the agent can invoke. Defaults to an empty
            registry (the agent has no tools).
    """

    def __init__(
        self,
        *,
        name: str,
        policy: Policy,
        system: str = "",
        tools: ToolRegistry | None = None,
    ) -> None:
        self.name = name
        self.policy = policy
        self.tools = tools if tools is not None else ToolRegistry()
        self.history: list[Message] = []
        if system:
            self.history.append(Message(role="system", content=system))

    def step(self, inbox: Sequence[Message] = ()) -> list[Message]:
        """One cycle. Returns assistant message + any tool result messages."""
        produced: list[Message] = []
        self.history.extend(inbox)

        assistant = self.policy.act(self.history, tools=self.tools.specs())
        if assistant.role != "assistant":
            raise ValueError(
                f"Policy for agent {self.name!r} returned role={assistant.role!r}; "
                "must be 'assistant'",
            )
        # Stamp the agent name onto outbound messages when the policy
        # didn't set one — useful for multi-agent transcripts.
        if assistant.name is None:
            assistant = Message(
                role=assistant.role,
                content=assistant.content,
                tool_calls=assistant.tool_calls,
                tool_call_id=assistant.tool_call_id,
                name=self.name,
            )
        self.history.append(assistant)
        produced.append(assistant)

        for call in assistant.tool_calls:
            result = self.tools.execute(call)
            self.history.append(result)
            produced.append(result)
        return produced

    def run(self, *, user_message: str, max_steps: int = 8) -> list[Message]:
        """Send a user message, step until no more tool calls (or limit).

        Loops until the most recent assistant message contains no tool
        calls (the policy has decided to "answer"), or ``max_steps`` is
        reached. Returns the full conversation history *including* prior
        turns from earlier ``run``/``step`` calls.
        """
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {max_steps}")

        produced = self.step([Message(role="user", content=user_message)])
        steps = 1
        while steps < max_steps:
            assistant = next((m for m in produced if m.role == "assistant"), None)
            if assistant is None or not assistant.tool_calls:
                break
            produced = self.step(())
            steps += 1
        return list(self.history)
