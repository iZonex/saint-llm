"""Tests for skill script execution."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from saint_llm_agents import ToolCall, ToolRegistry
from saint_llm_agents.skills import (
    ScriptResult,
    Skill,
    register_skill_scripts,
)


@dataclass
class _RecordingRunner:
    """ScriptRunner stub that captures invocations + returns programmable results."""

    stdout: str = ""
    stderr: str = ""
    ok: bool = True
    exit_code: int = 0
    calls: list[tuple[Path, list[str]]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def run(self, script_path: Path, args: Sequence[str]) -> ScriptResult:
        self.calls.append((script_path, list(args)))
        return ScriptResult(
            ok=self.ok,
            stdout=self.stdout,
            stderr=self.stderr,
            exit_code=self.exit_code,
        )


def _skill_with_scripts(tmp_path: Path, scripts: dict[str, str]) -> Skill:
    """Build a Skill on disk with the given script files."""
    root = tmp_path / "myskill"
    root.mkdir()
    (root / "SKILL.md").write_text(
        "---\nname: myskill\ndescription: test\n---\nbody",
        encoding="utf-8",
    )
    scripts_dir = root / "scripts"
    scripts_dir.mkdir()
    for filename, body in scripts.items():
        (scripts_dir / filename).write_text(body, encoding="utf-8")
    from saint_llm_agents.skills import load_skill  # noqa: PLC0415

    return load_skill(root)


def test_register_skill_scripts_returns_zero_when_no_scripts_dir(
    tmp_path: Path,
) -> None:
    skill = Skill(name="empty", description="x", instructions="body")
    registry = ToolRegistry()
    n = register_skill_scripts(registry, skill, _RecordingRunner())
    assert n == 0
    assert len(registry) == 0


def test_register_skill_scripts_creates_one_tool_per_file(tmp_path: Path) -> None:
    skill = _skill_with_scripts(
        tmp_path, {"hello.py": "print('hi')", "world.sh": "echo world"},
    )
    registry = ToolRegistry()
    n = register_skill_scripts(registry, skill, _RecordingRunner())
    assert n == 2
    assert "myskill_hello" in registry
    assert "myskill_world" in registry


def test_skipped_dotfiles(tmp_path: Path) -> None:
    skill = _skill_with_scripts(
        tmp_path, {".hidden": "secret", "real.py": "print('ok')"},
    )
    registry = ToolRegistry()
    n = register_skill_scripts(registry, skill, _RecordingRunner())
    assert n == 1
    assert "myskill_real" in registry
    assert "myskill_.hidden" not in registry


def test_tool_name_uses_default_skill_prefix(tmp_path: Path) -> None:
    skill = _skill_with_scripts(tmp_path, {"foo.py": "x"})
    registry = ToolRegistry()
    register_skill_scripts(registry, skill, _RecordingRunner())
    assert "myskill_foo" in registry


def test_tool_name_custom_prefix(tmp_path: Path) -> None:
    skill = _skill_with_scripts(tmp_path, {"foo.py": "x"})
    registry = ToolRegistry()
    register_skill_scripts(
        registry, skill, _RecordingRunner(), tool_name_prefix="custom_",
    )
    assert "custom_foo" in registry


def test_invoke_runs_script_with_flag_args(tmp_path: Path) -> None:
    skill = _skill_with_scripts(tmp_path, {"echo.py": "x"})
    runner = _RecordingRunner(stdout="ran echo")
    registry = ToolRegistry()
    register_skill_scripts(registry, skill, runner, arg_format="flags")
    msg = registry.execute(ToolCall(
        id="t", name="myskill_echo",
        arguments={"name": "world", "count": 3},
    ))
    assert msg.content == "ran echo"
    assert len(runner.calls) == 1
    script_path, argv = runner.calls[0]
    assert script_path.name == "echo.py"
    assert "--name" in argv
    assert "world" in argv
    assert "--count" in argv
    assert "3" in argv


def test_invoke_with_json_arg_format(tmp_path: Path) -> None:
    skill = _skill_with_scripts(tmp_path, {"echo.py": "x"})
    runner = _RecordingRunner(stdout="json mode")
    registry = ToolRegistry()
    register_skill_scripts(registry, skill, runner, arg_format="json")
    registry.execute(ToolCall(
        id="t", name="myskill_echo",
        arguments={"a": 1, "b": "hi"},
    ))
    _, argv = runner.calls[0]
    assert len(argv) == 1
    assert json.loads(argv[0]) == {"a": 1, "b": "hi"}


def test_invoke_handles_boolean_flags(tmp_path: Path) -> None:
    skill = _skill_with_scripts(tmp_path, {"echo.py": "x"})
    runner = _RecordingRunner()
    registry = ToolRegistry()
    register_skill_scripts(registry, skill, runner, arg_format="flags")
    registry.execute(ToolCall(
        id="t", name="myskill_echo",
        arguments={"verbose": True, "quiet": False},
    ))
    _, argv = runner.calls[0]
    assert "--verbose" in argv
    assert "--no-quiet" in argv


def test_invoke_converts_underscores_to_hyphens(tmp_path: Path) -> None:
    skill = _skill_with_scripts(tmp_path, {"echo.py": "x"})
    runner = _RecordingRunner()
    registry = ToolRegistry()
    register_skill_scripts(registry, skill, runner)
    registry.execute(ToolCall(
        id="t", name="myskill_echo",
        arguments={"out_file": "log.txt"},
    ))
    _, argv = runner.calls[0]
    assert "--out-file" in argv
    assert "log.txt" in argv


def test_failed_script_returns_error_string(tmp_path: Path) -> None:
    skill = _skill_with_scripts(tmp_path, {"echo.py": "x"})
    runner = _RecordingRunner(
        ok=False, exit_code=1, stderr="something broke",
    )
    registry = ToolRegistry()
    register_skill_scripts(registry, skill, runner)
    msg = registry.execute(ToolCall(
        id="t", name="myskill_echo", arguments={},
    ))
    assert msg.content.startswith("ERROR (exit 1)")
    assert "something broke" in msg.content


def test_unknown_arg_format_at_invoke_time_surfaces_as_tool_error(tmp_path: Path) -> None:
    """Bad arg_format is caught when the tool is invoked (not at register)."""
    skill = _skill_with_scripts(tmp_path, {"echo.py": "x"})
    registry = ToolRegistry()
    register_skill_scripts(
        registry, skill, _RecordingRunner(), arg_format="bogus",  # type: ignore[arg-type]
    )
    msg = registry.execute(ToolCall(
        id="t", name="myskill_echo", arguments={"a": 1},
    ))
    assert msg.content.startswith("ERROR:")
    assert "unknown arg_format" in msg.content


def test_in_memory_skill_with_no_root_returns_zero() -> None:
    """A Skill built in code (no root path) has no scripts to register."""
    skill = Skill(name="codeonly", description="x", instructions="...")
    registry = ToolRegistry()
    assert register_skill_scripts(registry, skill, _RecordingRunner()) == 0


def test_subdirectories_are_ignored(tmp_path: Path) -> None:
    """register_skill_scripts only registers files, not subdirectories."""
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text(
        "---\nname: s\ndescription: x\n---\nbody", encoding="utf-8",
    )
    scripts_dir = root / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "real.py").write_text("x", encoding="utf-8")
    (scripts_dir / "subdir").mkdir()
    (scripts_dir / "subdir" / "nested.py").write_text("x", encoding="utf-8")

    from saint_llm_agents.skills import load_skill  # noqa: PLC0415

    skill = load_skill(root)
    registry = ToolRegistry()
    n = register_skill_scripts(registry, skill, _RecordingRunner())
    assert n == 1


def test_compose_with_real_function_call_sandbox(tmp_path: Path) -> None:
    """Sanity: a thin adapter on FunctionCallSandbox satisfies ScriptRunner."""
    pytest.importorskip("saint_llm_sandbox")
    from saint_llm_sandbox import FunctionCallSandbox  # noqa: PLC0415

    skill = _skill_with_scripts(
        tmp_path,
        {"hello.py": "import sys; print('args:', sys.argv[1:])"},
    )
    sandbox = FunctionCallSandbox()

    class _Adapter:
        def run(self, script_path: Path, args: Sequence[str]) -> ScriptResult:
            res = sandbox.run(["python3", str(script_path), *args])
            return ScriptResult(
                ok=res.ok, stdout=res.stdout, stderr=res.stderr,
                exit_code=res.exit_code,
            )

    registry = ToolRegistry()
    register_skill_scripts(registry, skill, _Adapter())
    msg = registry.execute(ToolCall(
        id="t", name="myskill_hello", arguments={"who": "world"},
    ))
    assert "args:" in msg.content
