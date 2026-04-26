"""SkillRegistry — name->Skill index + tool-registration helpers.

Two ways to surface a skill to the agent:

* **Catalog string** — :meth:`SkillRegistry.catalog` produces a short
  block listing each skill's name + description. Drop it into the
  system prompt so the model knows what's available without paying
  the token cost of every full body.
* **On-demand tool** — :func:`register_skills_as_tools` registers one
  ``use_<skill_name>`` tool per skill that returns the skill body
  when the agent calls it. The model picks one, gets the
  instructions, then proceeds.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from saint_llm_agents.skills.loader import Skill
from saint_llm_agents.tool import FunctionTool, ToolRegistry, ToolSpec


class SkillRegistry:
    """Name -> :class:`Skill` map with prompt + tool helpers."""

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills: dict[str, Skill] = {}
        for s in skills:
            self.add(s)

    def add(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"skill {skill.name!r} already registered")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"skill {name!r} not registered")
        return self._skills[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._skills.keys())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterable[Skill]:
        return iter(self._skills.values())

    def catalog(self, *, header: str = "Available skills:") -> str:
        """Render a short prompt-ready catalog of skill name + description.

        Returns an empty string when the registry has no skills, so the
        caller can splice it into a prompt unconditionally.
        """
        if not self._skills:
            return ""
        lines = [header]
        for skill in self._skills.values():
            desc = skill.description or "(no description)"
            lines.append(f"- {skill.name}: {desc}")
        return "\n".join(lines)


def register_skills_as_tools(
    registry: ToolRegistry,
    skills: Sequence[Skill] | SkillRegistry,
    *,
    tool_name_prefix: str = "use_",
) -> None:
    """Expose each skill as a ``use_<skill_name>`` tool in ``registry``.

    Calling the tool returns the skill body (the Markdown instructions).
    The agent can then read it and proceed. ``tool_name_prefix`` lets
    callers override the default ``"use_"`` prefix.
    """
    for skill in skills:
        registry.register(_skill_to_tool(skill, prefix=tool_name_prefix))


def _skill_to_tool(skill: Skill, *, prefix: str) -> FunctionTool:
    spec = ToolSpec(
        name=f"{prefix}{skill.name}",
        description=(
            f"Load instructions for the {skill.name!r} skill. "
            f"Description: {skill.description}"
        ).strip(),
        parameters={"type": "object", "properties": {}, "required": []},
    )
    instructions = skill.instructions

    def _call(**_: Any) -> str:
        return instructions

    return FunctionTool(spec=spec, fn=_call)
