"""Run a skill's executable scripts as agent tools.

A skill's directory may contain a ``scripts/`` subdirectory with
helper executables (Python files, shell scripts, etc.) the skill's
instructions reference. By default the agent runtime can only read
the skill body — it has no way to execute those scripts.

This module bridges the gap with a :class:`ScriptRunner` Protocol:
any caller-supplied object that knows how to ``run(script_path,
args)`` (typically a sandbox wrapper) becomes the execution engine.
:func:`register_skill_scripts` discovers every executable under
``skill.scripts_dir`` and registers each as a :class:`FunctionTool`
in the agent's :class:`ToolRegistry`. Calling the tool encodes the
agent's keyword arguments into ``--key value`` flags (or a JSON
blob), passes them to the runner, and returns stdout.

The Protocol-based design keeps the ``agents`` package free of
``saint_llm_sandbox`` as a hard dependency — users plug whichever
sandbox tier they want by adapting it to :class:`ScriptRunner`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from saint_llm_agents.skills.loader import Skill
from saint_llm_agents.tool import FunctionTool, ToolRegistry, ToolSpec

ArgFormat = Literal["flags", "json"]


@dataclass(frozen=True)
class ScriptResult:
    """Subset of :class:`SandboxResult` we actually consume."""

    ok: bool
    stdout: str
    stderr: str = ""
    exit_code: int = 0


class ScriptRunner(Protocol):
    """Executes a script file with arguments and returns the result.

    Sandbox wrappers, raw subprocess callers, or stub mocks all fit
    this Protocol. The ``run`` method is the only required entry
    point.
    """

    def run(
        self, script_path: Path, args: Sequence[str],
    ) -> ScriptResult: ...


def register_skill_scripts(
    registry: ToolRegistry,
    skill: Skill,
    runner: ScriptRunner,
    *,
    arg_format: ArgFormat = "flags",
    tool_name_prefix: str | None = None,
) -> int:
    """Register each script in ``skill.scripts_dir`` as a tool.

    Args:
        registry:         the :class:`ToolRegistry` to populate.
        skill:            a loaded :class:`Skill`. If
            ``skill.scripts_dir`` is None, this is a no-op.
        runner:           an object satisfying :class:`ScriptRunner`.
        arg_format:       how to convert tool kwargs into argv:

            * ``"flags"`` (default) — each kwarg ``key=value`` becomes
              ``--key value`` in argv. Booleans use ``--key`` /
              ``--no-key`` form.
            * ``"json"`` — argv is a single positional argument
              containing ``json.dumps(kwargs)``. The script is
              expected to ``json.loads(sys.argv[1])``.
        tool_name_prefix: prefix prepended to each tool name. Defaults
            to ``f"{skill.name}_"`` when None.

    Returns:
        the count of scripts registered. Zero when the skill has no
        ``scripts/`` directory or it's empty.
    """
    if skill.scripts_dir is None:
        return 0

    prefix = (
        tool_name_prefix
        if tool_name_prefix is not None
        else f"{skill.name}_"
    )

    count = 0
    for script_path in sorted(skill.scripts_dir.iterdir()):
        if not script_path.is_file():
            continue
        if script_path.name.startswith("."):
            continue
        registry.register(
            _build_script_tool(
                runner=runner,
                script_path=script_path,
                tool_name=f"{prefix}{script_path.stem}",
                skill_name=skill.name,
                arg_format=arg_format,
            ),
        )
        count += 1
    return count


def _build_script_tool(
    *,
    runner: ScriptRunner,
    script_path: Path,
    tool_name: str,
    skill_name: str,
    arg_format: ArgFormat,
) -> FunctionTool:
    spec = ToolSpec(
        name=tool_name,
        description=(
            f"Run the {script_path.name!r} script from skill "
            f"{skill_name!r}. Arguments are forwarded as "
            f"{'--key value flags' if arg_format == 'flags' else 'a single JSON positional'}."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
    )

    def _call(**kwargs: Any) -> str:
        argv = _kwargs_to_argv(kwargs, arg_format=arg_format)
        result = runner.run(script_path, argv)
        if not result.ok:
            stderr = result.stderr.strip()
            return f"ERROR (exit {result.exit_code}): {stderr or result.stdout}"
        return result.stdout

    return FunctionTool(spec=spec, fn=_call)


def _kwargs_to_argv(
    kwargs: dict[str, Any], *, arg_format: ArgFormat,
) -> list[str]:
    if arg_format == "json":
        return [json.dumps(kwargs)]
    if arg_format == "flags":
        argv: list[str] = []
        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            if value is True:
                argv.append(flag)
            elif value is False:
                argv.append(f"--no-{key.replace('_', '-')}")
            else:
                argv.append(flag)
                argv.append(str(value))
        return argv
    raise ValueError(f"unknown arg_format {arg_format!r}")
