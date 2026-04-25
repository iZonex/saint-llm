"""Saint LLM sandbox SDK (libdsec equivalent).

Wraps DSec (DeepSeek Elastic Compute equivalent) — Rust services in `packages/sandbox/dsec/`.

Substrates:
    function_call  — Stateless invocation, warm container pool
    container      — Docker + EROFS layered loading from 3FS
    microvm        — Firecracker + overlaybd CoW
    fullvm         — QEMU for arbitrary guest OS

All four expose the same interface (command exec, file transfer, TTY).
"""

__version__ = "0.0.1"
