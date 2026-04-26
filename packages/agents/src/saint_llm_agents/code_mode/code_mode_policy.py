"""Code Mode Policy adapter — wraps any base Policy so it speaks Python.

ADR-0013: instead of emitting JSON tool-call envelopes, the model
writes a Python snippet that uses the ``tools`` namespace. This wrapper
handles the protocol bookkeeping:

1. Inject the API stub (from :mod:`tool_to_api`) into the prompt as
   a system message before the actor runs.
2. Receive the actor's reply (a Python snippet).
3. Execute via :class:`Executor`.
4. Translate the execution result back into ``Message`` shape that
   the standard agent loop understands — returning a synthetic
   assistant message that summarizes the result.

The agent runtime sees a normal :class:`Policy`; Code Mode is purely
an internal detail of this wrapper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from saint_llm_agents.code_mode.sandbox_executor import (
    ExecutionResult,
    Executor,
    InProcessExecutor,
)
from saint_llm_agents.code_mode.tool_to_api import generate_api_stub
from saint_llm_agents.message import Message
from saint_llm_agents.policy import Policy
from saint_llm_agents.tool import ToolRegistry, ToolSpec


@dataclass
class CodeModePolicy:
    """Wrap a base :class:`Policy` to use Code Mode for tool calls.

    Args:
        base_policy: the underlying actor (SaintLLM, OpenAI, mock).
            The wrapper expects ``base_policy.act`` to return an
            assistant message whose ``content`` is a Python snippet.
        registry:    tools the snippet can call.
        executor:    sandbox runner. Default :class:`InProcessExecutor`
            for tests; production swaps in a microvm/container backend.
        api_namespace: name of the wrapping class injected into prompts.
            Default ``"tools"`` so the model writes
            ``tools.<name>(...)``.
        prompt_prefix: optional system text prepended above the API
            stub when injecting it into messages. Default mentions
            "Use Code Mode: write Python that calls tools.<name>(...).
            Assign your final answer to RESULT."
    """

    base_policy: Policy
    registry: ToolRegistry
    executor: Executor = None  # type: ignore[assignment]
    api_namespace: str = "tools"
    prompt_prefix: str = (
        "You are operating in Code Mode. Write a short Python snippet "
        "that calls the tools API below. Do NOT explain — output only "
        "executable Python. Assign your final answer to ``RESULT``.\n\n"
    )

    def __post_init__(self) -> None:
        if self.executor is None:
            self.executor = InProcessExecutor()

    def _inject_api_stub(self, messages: Sequence[Message]) -> list[Message]:
        """Prepend a system message with the tools API stub."""
        stub = generate_api_stub(list(self.registry.specs()), namespace=self.api_namespace)
        api_text = self.prompt_prefix + "```python\n" + stub.source + "\n```"
        return [Message(role="system", content=api_text), *messages]

    def act(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),  # ignored — Code Mode injects via prompt
    ) -> Message:
        """Run the base policy with API stub injected, execute the
        result, and return a synthesized assistant message.

        The returned message has ``content`` describing what happened
        (return value or error), and ``tool_calls`` populated from
        the executor's per-call log.
        """
        del tools  # the agent runtime hands us the registry's specs but
        # Code Mode injects them as a Python namespace, not a list.

        wrapped = self._inject_api_stub(messages)
        actor_msg = self.base_policy.act(wrapped, tools=())
        if actor_msg.role != "assistant":
            raise ValueError(
                f"Code Mode base policy must return assistant; got {actor_msg.role!r}",
            )
        snippet = actor_msg.content
        result = self.executor.execute(snippet, registry=self.registry)
        return self._result_to_message(result)

    def _result_to_message(self, result: ExecutionResult) -> Message:
        """Translate an execution result into the assistant message shape.

        Tool-calls captured during execution are NOT returned as
        :class:`ToolCall` objects (Code Mode bypasses that envelope by
        design); they're recorded in the message's ``content`` and
        ``name`` field for transcript visibility.
        """
        if result.ok:
            content = (
                f"Code Mode result: {result.return_value!r}"
                if result.return_value is not None
                else "Code Mode result: (no RESULT assigned)"
            )
            if result.stdout:
                content += f"\n[stdout]\n{result.stdout}"
        else:
            content = (
                f"Code Mode error: {result.error_type}: {result.error_message}"
            )
            if result.stderr:
                content += f"\n[stderr]\n{result.stderr}"
        return Message(role="assistant", content=content, name="code_mode")
