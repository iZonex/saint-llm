"""Policy abstraction — the "brain" that turns history → next assistant message.

A :class:`Policy` is *anything* that can read a conversation history and
emit one assistant :class:`Message` (possibly carrying tool calls). The
runtime treats it as a black box, which means the same agent code runs
unchanged whether the brain is:

* our trained SaintLLM (eventually),
* an external API client (OpenAI / Anthropic / local Ollama),
* a deterministic mock for tests,
* a hard-coded rule engine.

Two ready-to-use implementations:

* :class:`MockPolicy` — replays a fixed list of pre-built responses in
  order. The default test harness for the rest of the agent code.
* :class:`CallablePolicy` — wraps any function ``(messages, *, tools) ->
  Message`` so arbitrary policies plug in without subclassing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from saint_llm_agents.message import Message
from saint_llm_agents.tool import ToolSpec


class Policy(Protocol):
    """Read history, emit next assistant message.

    Implementations must always return a :class:`Message` with
    ``role == "assistant"``. Tool invocations live in
    :attr:`Message.tool_calls`; the runtime — not the policy — actually
    runs them.
    """

    def act(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
    ) -> Message: ...


class CallablePolicy:
    """Adapt any callable into a :class:`Policy`.

    Useful for closures and one-off policies in tests::

        def echo(messages, *, tools):
            return Message(role="assistant", content=messages[-1].content)
        policy = CallablePolicy(echo)
    """

    def __init__(self, fn: Callable[..., Message]) -> None:
        self._fn = fn

    def act(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
    ) -> Message:
        return self._fn(messages, tools=tools)


class MockPolicy:
    """Replay a fixed list of pre-built responses in order.

    Each :meth:`act` returns the next response and advances an internal
    cursor. When the list is exhausted, raises :class:`StopIteration` so
    over-long agent loops blow up loudly instead of silently spinning.
    """

    def __init__(self, responses: Sequence[Message]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def act(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
    ) -> Message:
        if self._idx >= len(self._responses):
            raise StopIteration("MockPolicy exhausted")
        out = self._responses[self._idx]
        self._idx += 1
        return out

    @property
    def remaining(self) -> int:
        return len(self._responses) - self._idx
