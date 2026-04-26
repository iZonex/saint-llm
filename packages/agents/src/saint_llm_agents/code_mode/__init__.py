"""Code Mode runtime for tool calls (ADR-0013, AGT-01).

Replaces the DSML XML schema (rejected per ADR-0013): the model writes
Python that calls ``tools.<name>(...)``; the runtime executes in a
sandbox; results return as structured assistant messages.

Modules:
    tool_to_api      — generate the ``tools`` Python API stub from a ToolRegistry
    sandbox_executor — Executor Protocol + InProcessExecutor (tests / dev)
    code_mode_policy — Policy wrapper that injects the stub, runs the actor,
                       executes the snippet, returns a result message
"""

from saint_llm_agents.code_mode.code_mode_policy import CodeModePolicy
from saint_llm_agents.code_mode.sandbox_executor import (
    ExecutionResult,
    Executor,
    InProcessExecutor,
)
from saint_llm_agents.code_mode.tool_to_api import ApiStub, generate_api_stub

__all__ = [
    "ApiStub",
    "CodeModePolicy",
    "ExecutionResult",
    "Executor",
    "InProcessExecutor",
    "generate_api_stub",
]
