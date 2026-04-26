"""Anthropic Skills loader for saint-llm agents.

A *skill* is a self-contained directory packaging instructions and
(optionally) scripts that an agent can invoke. The discovery convention
mirrors Anthropic's spec:

* one subdirectory per skill;
* each subdirectory contains a ``SKILL.md`` file with YAML-style
  frontmatter (``name``, ``description``, optional ``license``,
  ``allowed-tools``) followed by a Markdown body of instructions;
* optional ``scripts/`` directory with executable helpers the skill
  refers to in its instructions.

This loader:

* :func:`load_skills` discovers + parses skills under a root directory.
* :class:`SkillRegistry` indexes them by name and renders a catalog
  string for the agent's system prompt.
* :func:`register_skills_as_tools` exposes each skill as a
  :class:`FunctionTool` named ``use_<skill_name>`` whose ``__call__``
  returns the skill body — letting the agent pull a skill on demand
  without bloating the system prompt with every skill at once.
"""

from saint_llm_agents.skills.loader import (
    Skill,
    SkillLoadError,
    load_skill,
    load_skills,
)
from saint_llm_agents.skills.registry import (
    SkillRegistry,
    register_skills_as_tools,
)
from saint_llm_agents.skills.scripts import (
    ScriptResult,
    ScriptRunner,
    register_skill_scripts,
)

__all__ = [
    "ScriptResult",
    "ScriptRunner",
    "Skill",
    "SkillLoadError",
    "SkillRegistry",
    "load_skill",
    "load_skills",
    "register_skill_scripts",
    "register_skills_as_tools",
]
