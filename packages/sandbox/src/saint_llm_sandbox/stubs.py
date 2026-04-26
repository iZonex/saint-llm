"""Stub implementations for container / microvm / fullvm sandboxes.

Each stub raises ``NotImplementedError`` from :meth:`run` /
:meth:`run_python`. They exist so the Sandbox Protocol shape is
established for callers that switch tiers via dependency injection;
production deploy plugs in the real Rust DSec services here.

When the real backends land, replace this module's contents per
substrate; the public Protocol surface stays unchanged.
"""

from __future__ import annotations

from saint_llm_sandbox.result import SandboxResult
from saint_llm_sandbox.sandbox import SandboxConfig


class _UnimplementedSandbox:
    """Common base — raises NotImplementedError on any run() call."""

    name: str = "unimplemented"

    def run(
        self,
        argv: list[str],
        *,
        stdin: bytes = b"",
        config: SandboxConfig | None = None,
    ) -> SandboxResult:
        raise NotImplementedError(
            f"{self.name} sandbox is not yet implemented; "
            "use FunctionCallSandbox for v0.0 trusted-code paths.",
        )

    def run_python(
        self,
        code: str,
        *,
        config: SandboxConfig | None = None,
    ) -> SandboxResult:
        raise NotImplementedError(
            f"{self.name} sandbox is not yet implemented; "
            "use FunctionCallSandbox for v0.0 trusted-code paths.",
        )


class ContainerSandbox(_UnimplementedSandbox):
    """Docker + EROFS layered loading. Stub at v0.0 / production TBD.

    Will use Rust DSec service in ``packages/sandbox/dsec/`` when wired.
    """

    name = "container"


class MicroVMSandbox(_UnimplementedSandbox):
    """Firecracker + overlaybd CoW. Stub at v0.0 / production TBD."""

    name = "microvm"


class FullVMSandbox(_UnimplementedSandbox):
    """QEMU full-VM. Stub at v0.0 / production TBD."""

    name = "fullvm"
