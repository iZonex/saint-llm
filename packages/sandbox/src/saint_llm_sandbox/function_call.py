"""FunctionCallSandbox — warm Python pool, no isolation.

The lightest tier: spawn a fresh Python subprocess per call (or
re-use a warmed interpreter) with timeout + memory caps. Trust
level **HIGH** — the caller must already trust the code; this
backend protects against accidents (timeouts, OOM) but NOT against
intentional sandbox escape.

Use this tier for:
* Code Mode tools execution by our own model in trusted contexts
* Test-only runs
* Fast prototyping

For untrusted code use :class:`saint_llm_sandbox.microvm.MicroVMSandbox`
(stub at v0.0; real impl when Firecracker is wired).
"""

from __future__ import annotations

import contextlib
import os
import resource
import subprocess
import time
from pathlib import Path

from saint_llm_sandbox.result import SandboxResult
from saint_llm_sandbox.sandbox import SandboxConfig


class FunctionCallSandbox:
    """In-process / subprocess Python sandbox.

    ``run(argv)`` spawns a subprocess with timeout + memory caps.
    ``run_python(code)`` spawns ``python -c <code>``.

    Memory cap uses ``resource.setrlimit(RLIMIT_AS, ...)`` on
    POSIX; on platforms without that primitive (Windows), the
    setting is a no-op and the timeout remains the only cap.
    """

    def __init__(self, *, python_exe: str | None = None) -> None:
        self._python = python_exe if python_exe is not None else self._default_python()

    @staticmethod
    def _default_python() -> str:
        # Prefer the active interpreter. Fall back to "python3" on PATH.
        import sys  # noqa: PLC0415

        return sys.executable or "python3"

    def run(
        self,
        argv: list[str],
        *,
        stdin: bytes = b"",
        config: SandboxConfig | None = None,
    ) -> SandboxResult:
        cfg = config if config is not None else SandboxConfig()
        env = dict(os.environ) if not cfg.env else dict(cfg.env)
        cwd = Path(cfg.cwd) if cfg.cwd else None

        def _set_limits() -> None:
            # Best-effort memory cap (POSIX). Some platforms (macOS,
            # certain containers) reject AS limits — suppress and
            # rely on the timeout instead.
            mem_bytes = max(1, cfg.memory_mb) * 1024 * 1024
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(
                    resource.RLIMIT_AS, (mem_bytes, mem_bytes),
                )
            with contextlib.suppress(ValueError, OSError):
                # Disable core dumps so OOMs don't leave large files.
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        started = time.perf_counter()
        try:
            proc = subprocess.run(
                argv,
                input=stdin,
                capture_output=True,
                env=env,
                cwd=str(cwd) if cwd else None,
                timeout=cfg.timeout_s,
                check=False,
                preexec_fn=_set_limits if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                ok=False,
                exit_code=-1,
                stdout=(exc.stdout or b"").decode("utf-8", errors="replace"),
                stderr=(exc.stderr or b"").decode("utf-8", errors="replace"),
                duration_s=time.perf_counter() - started,
                error=f"timeout after {cfg.timeout_s}s",
            )
        except FileNotFoundError as exc:
            return SandboxResult(
                ok=False,
                exit_code=-1,
                duration_s=time.perf_counter() - started,
                error=f"executable not found: {exc}",
            )

        return SandboxResult(
            ok=proc.returncode == 0,
            exit_code=proc.returncode,
            stdout=proc.stdout.decode("utf-8", errors="replace"),
            stderr=proc.stderr.decode("utf-8", errors="replace"),
            duration_s=time.perf_counter() - started,
        )

    def run_python(
        self,
        code: str,
        *,
        config: SandboxConfig | None = None,
    ) -> SandboxResult:
        return self.run([self._python, "-c", code], config=config)
