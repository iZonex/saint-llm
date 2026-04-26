"""Sandbox Protocol — uniform interface across all four substrates.

Backends:

* :class:`saint_llm_sandbox.function_call.FunctionCallSandbox` —
  warm Python interpreter pool; in-process. Trust-level: HIGH (no
  isolation against intentionally-malicious code).
* :class:`saint_llm_sandbox.container.ContainerSandbox` — Docker /
  EROFS layered loading. Stub at v0.0; real impl when Rust DSec
  services are wired.
* :class:`saint_llm_sandbox.microvm.MicroVMSandbox` — Firecracker +
  overlaybd CoW. Stub at v0.0.
* :class:`saint_llm_sandbox.fullvm.FullVMSandbox` — QEMU full-VM
  for arbitrary guest OS. Stub at v0.0.

All backends accept the same :class:`SandboxConfig` and return
:class:`SandboxResult`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from saint_llm_sandbox.result import SandboxResult


@dataclass(frozen=True)
class SandboxConfig:
    """Common knobs every sandbox backend honors.

    Attributes:
        timeout_s:   wall-clock cap on one invocation. Default 30s.
        memory_mb:   peak memory cap (best-effort). Default 1024.
        cpu_cores:   max CPU cores. Default 1.
        env:         environment variables passed in.
        cwd:         working directory inside the sandbox (backend may
            map this to a tmpdir).
        readonly:    when True, the sandbox FS is read-only after
            mount; the run cannot write outside its tmp.
        network:     when False, network is disabled.
    """

    timeout_s: float = 30.0
    memory_mb: int = 1024
    cpu_cores: int = 1
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str = "/tmp"
    readonly: bool = True
    network: bool = False


class Sandbox(Protocol):
    """Generic sandbox: run a command, return a result.

    Implementations are stateless across runs by default; pooled
    backends (function_call) hold a warm interpreter behind the
    scenes but each :meth:`run` is observably independent.
    """

    def run(
        self,
        argv: list[str],
        *,
        stdin: bytes = b"",
        config: SandboxConfig | None = None,
    ) -> SandboxResult: ...

    def run_python(
        self,
        code: str,
        *,
        config: SandboxConfig | None = None,
    ) -> SandboxResult: ...
