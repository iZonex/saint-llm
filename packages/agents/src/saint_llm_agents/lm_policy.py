"""TextLMPolicy — bridge any text generator into the Agent runtime.

Built-in :class:`Policy` implementations (:class:`MockPolicy`,
:class:`CallablePolicy`) build :class:`Message` objects directly. For
real-LM-backed agents the generator outputs *text* instead, and the
runtime needs to:

1. Render the conversation history into a model prompt (chat template).
2. Generate the assistant's response.
3. Parse any tool-call markers out of the text.
4. Return a structured :class:`Message`.

This module ships :class:`TextLMPolicy`: the caller provides a
``generate(history) -> str`` callable that handles steps 1 + 2; the
Policy handles step 3 + 4 via :func:`parse_tool_calls` and emits a
properly-shaped assistant :class:`Message` with both visible content
and structured ``tool_calls``.

Crucially this does *not* require :mod:`saint_llm_inference` or any
ML-side dependency — the agent runtime stays light. Users compose
their own generation pipeline (using :class:`ChatSession`,
:func:`top_p_sample`, an external API, etc.) and hand it as the
``generate`` callable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from saint_llm_agents.message import Message
from saint_llm_agents.tool import ToolSpec
from saint_llm_agents.tool_call_parser import parse_tool_calls

GenerateFn = Callable[[Sequence[Message]], str]
ParserFormat = Literal["xml", "json", "auto", "none"]


class TextLMPolicy:
    """Policy that wraps a text-generating callable.

    Args:
        generate:        ``(history) -> str``. Receives the agent's
            conversation history and returns the assistant's raw
            output text.
        parse_format:    how to extract tool calls from the output.
            ``"auto"`` (default) tries XML and JSON-fenced markers;
            ``"xml"`` / ``"json"`` restrict to one format;
            ``"none"`` skips parsing — the resulting Message will
            have ``tool_calls=()``.
        strict:          if True, malformed tool-call blocks raise
            instead of being silently dropped.

    Example::

        def generate(history):
            session = ChatSession(model=..., tokenizer=...)
            for turn in history:
                if turn.role == "user":
                    session.add_user(turn.content)
                # ... etc
            return session.respond()

        policy = TextLMPolicy(generate)
        agent = Agent(name="bot", policy=policy, tools=registry)
        agent.run(user_message="what's the weather?")
    """

    def __init__(
        self,
        generate: GenerateFn,
        *,
        parse_format: ParserFormat = "auto",
        strict: bool = False,
    ) -> None:
        self._generate = generate
        self._fmt = parse_format
        self._strict = strict

    def act(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
    ) -> Message:
        """Generate, parse tool calls, return a structured assistant message."""
        del tools  # caller has already encoded available tools into the prompt
        text = self._generate(messages)
        if not isinstance(text, str):
            raise TypeError(
                f"generate(history) must return str; got {type(text).__name__}",
            )
        if self._fmt == "none":
            return Message(role="assistant", content=text)
        tool_calls = parse_tool_calls(text, fmt=self._fmt, strict=self._strict)
        return Message(
            role="assistant",
            content=text,
            tool_calls=tuple(tool_calls),
        )
