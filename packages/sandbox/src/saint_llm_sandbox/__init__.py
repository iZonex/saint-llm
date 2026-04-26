"""Saint LLM sandbox SDK (Workstream C2 — sandbox tiers).

Wraps DSec (DeepSeek Elastic Compute equivalent) — Rust services in
`packages/sandbox/dsec/` (planned).

Substrates:
    function_call  — Warm Python pool / subprocess. SHIPPED v0.0.
    container      — Docker + EROFS layered loading from 3FS. STUB.
    microvm        — Firecracker + overlaybd CoW. STUB.
    fullvm         — QEMU for arbitrary guest OS. STUB.

All four expose the same :class:`Sandbox` Protocol (command exec via
``run()`` and ``run_python()``) so callers swap tiers via dependency
injection. Trust levels:

    function_call: HIGH-trust (no isolation against malicious code)
    container:     MEDIUM-trust (Docker isolation, kernel shared)
    microvm:       LOW-trust (Firecracker, dedicated kernel per call)
    fullvm:        LOWEST-trust (QEMU, full guest OS)
"""

from saint_llm_sandbox.function_call import FunctionCallSandbox
from saint_llm_sandbox.result import SandboxResult
from saint_llm_sandbox.sandbox import Sandbox, SandboxConfig
from saint_llm_sandbox.stubs import (
    ContainerSandbox,
    FullVMSandbox,
    MicroVMSandbox,
)

__version__ = "0.0.1"

__all__ = [
    "ContainerSandbox",
    "FullVMSandbox",
    "FunctionCallSandbox",
    "MicroVMSandbox",
    "Sandbox",
    "SandboxConfig",
    "SandboxResult",
]
