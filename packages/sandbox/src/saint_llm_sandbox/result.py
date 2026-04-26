"""Common result type for all sandbox tiers.

Every backend (function_call / container / microvm / fullvm) returns
a :class:`SandboxResult` so the calling agent / Code Mode runtime
sees a uniform shape regardless of substrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of one sandboxed invocation.

    Attributes:
        ok:           True iff the process exited cleanly (exit_code == 0)
            and there were no setup / timeout errors.
        exit_code:    process exit code; -1 when the run never reached
            a process (e.g. spawn failed, sandbox unavailable).
        stdout:       captured stdout (utf-8 decoded).
        stderr:       captured stderr (utf-8 decoded).
        duration_s:   wall time from spawn to completion / kill.
        error:        free-form error text when ``ok=False`` for non-
            process reasons (timeout, OOM, sandbox unavailable).
        files:        produced files keyed by path (relative to the
            sandbox CWD). Optional; backends that don't track output
            files leave this empty.
    """

    ok: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    error: str | None = None
    files: dict[str, bytes] = field(default_factory=dict)
