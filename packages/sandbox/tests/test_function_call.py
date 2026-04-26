"""Tests for FunctionCallSandbox + stub backends."""

from __future__ import annotations

import pytest
from saint_llm_sandbox import (
    ContainerSandbox,
    FullVMSandbox,
    FunctionCallSandbox,
    MicroVMSandbox,
    SandboxConfig,
)


def test_function_call_runs_python_code() -> None:
    sb = FunctionCallSandbox()
    result = sb.run_python("print('hello')")
    assert result.ok
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_function_call_captures_stderr() -> None:
    sb = FunctionCallSandbox()
    result = sb.run_python("import sys; sys.stderr.write('warn')")
    assert "warn" in result.stderr


def test_function_call_nonzero_exit_marks_not_ok() -> None:
    sb = FunctionCallSandbox()
    result = sb.run_python("import sys; sys.exit(7)")
    assert not result.ok
    assert result.exit_code == 7


def test_function_call_timeout_returns_error() -> None:
    sb = FunctionCallSandbox()
    cfg = SandboxConfig(timeout_s=0.5)
    result = sb.run_python("import time; time.sleep(5)", config=cfg)
    assert not result.ok
    assert result.error is not None
    assert "timeout" in result.error.lower()


def test_function_call_unknown_executable_returns_error() -> None:
    sb = FunctionCallSandbox(python_exe="/nonexistent/python")
    result = sb.run_python("print('x')")
    assert not result.ok
    assert result.error is not None
    assert "not found" in result.error.lower()


def test_function_call_run_argv_directly() -> None:
    sb = FunctionCallSandbox()
    result = sb.run([sb._python, "-c", "print(2+2)"])  # type: ignore[attr-defined]
    assert result.ok
    assert "4" in result.stdout


def test_function_call_records_duration() -> None:
    sb = FunctionCallSandbox()
    result = sb.run_python("print(1)")
    assert result.duration_s >= 0.0


def test_sandbox_config_defaults() -> None:
    cfg = SandboxConfig()
    assert cfg.timeout_s == 30.0
    assert cfg.memory_mb == 1024
    assert cfg.network is False
    assert cfg.readonly is True


def test_container_sandbox_stub_raises() -> None:
    sb = ContainerSandbox()
    with pytest.raises(NotImplementedError, match="container"):
        sb.run(["echo", "hi"])
    with pytest.raises(NotImplementedError, match="container"):
        sb.run_python("print(1)")


def test_microvm_sandbox_stub_raises() -> None:
    sb = MicroVMSandbox()
    with pytest.raises(NotImplementedError, match="microvm"):
        sb.run_python("print(1)")


def test_fullvm_sandbox_stub_raises() -> None:
    sb = FullVMSandbox()
    with pytest.raises(NotImplementedError, match="fullvm"):
        sb.run_python("print(1)")
