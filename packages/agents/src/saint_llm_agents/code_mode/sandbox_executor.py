"""Sandboxed Python execution for Code Mode (ADR-0013).

The model emits Python that calls a ``tools`` namespace; this executor
runs that snippet in a controlled scope where ``tools.<name>(...)``
actually dispatches into the :class:`ToolRegistry`. Captured results
(return values, prints, exceptions) are packaged into a structured
``ExecutionResult`` for the agent loop.

v0.0 ships an **in-process** executor: simple, no isolation, suitable
for trusted code paths and tests. The full sandbox tiers (microvm /
container / fullvm via ``saint-llm-sandbox``) plug in here later by
implementing the same :class:`Executor` Protocol.
"""

from __future__ import annotations

import contextlib
import io
import traceback
from dataclasses import dataclass, field
from typing import Any, Protocol

from saint_llm_agents.tool import ToolRegistry


@dataclass
class ExecutionResult:
    """Captured outcome of one Code Mode snippet."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    error_type: str | None = None
    error_message: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class Executor(Protocol):
    """Contract for any Code Mode executor backend."""

    def execute(self, code: str, *, registry: ToolRegistry) -> ExecutionResult: ...


class _ToolNamespace:
    """Runtime-bound replacement for the ``tools`` class in generated code.

    Looking up ``tools.<name>`` returns a callable that invokes the tool
    and records the call. ``__getattr__`` raises a clear error for
    non-registered names so the sandboxed code surfaces typos as
    runtime exceptions.
    """

    def __init__(self, registry: ToolRegistry, log: list[dict[str, Any]]) -> None:
        self._registry = registry
        self._log = log

    def __getattr__(self, name: str) -> Any:
        if name not in self._registry:
            raise AttributeError(f"tool {name!r} is not registered")
        tool = self._registry.get(name)

        def _call(**kwargs: Any) -> Any:
            self._log.append({"name": name, "arguments": dict(kwargs)})
            return tool(**kwargs)

        return _call


class InProcessExecutor:
    """Trivial executor — runs the snippet via ``exec`` in a fresh scope.

    Provides the ``tools`` namespace and a ``RESULT`` slot the snippet
    can assign to for explicit return; any uncaught exception is caught
    and surfaced via :class:`ExecutionResult.error_*`.

    NOT a security boundary. Use only for trusted code (tests, dev).
    Production deployments should swap in a microvm/container backend
    that implements the :class:`Executor` Protocol.
    """

    def execute(self, code: str, *, registry: ToolRegistry) -> ExecutionResult:
        log: list[dict[str, Any]] = []
        ns = _ToolNamespace(registry, log)
        scope: dict[str, Any] = {"tools": ns, "RESULT": None}
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, scope)
            return ExecutionResult(
                ok=True,
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
                return_value=scope.get("RESULT"),
                tool_calls=log,
            )
        except Exception as exc:
            return ExecutionResult(
                ok=False,
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue() + traceback.format_exc(),
                error_type=type(exc).__name__,
                error_message=str(exc),
                tool_calls=log,
            )
