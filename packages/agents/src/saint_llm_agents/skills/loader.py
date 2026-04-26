"""Discover + parse Anthropic-style skill directories.

A skill directory ``my-skill/`` contains:

* ``SKILL.md`` — frontmatter + Markdown body.
* ``scripts/`` (optional) — executable helpers referenced from the body.

Frontmatter format (YAML-shaped, parsed by a minimal in-house parser
sufficient for the documented fields — string scalars and string
lists). Example::

    ---
    name: pdf-extractor
    description: Extract structured fields from PDF invoices.
    license: MIT
    allowed-tools:
      - read_file
      - run_python
    ---

    The skill body in Markdown. The agent reads this when it decides
    to use this skill.

Avoids a hard dependency on PyYAML — frontmatter for skills is
restricted to simple flat key/value entries plus string lists. If we
ever need general YAML (anchors, nested mappings, etc.) we'll switch
to ``yaml.safe_load`` then.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

SKILL_FILE_NAME = "SKILL.md"
_FRONTMATTER_DELIM = "---"


class SkillLoadError(ValueError):
    """Raised when a SKILL.md file is malformed."""


@dataclass(frozen=True)
class Skill:
    """One parsed skill.

    Attributes:
        name:           unique skill identifier (frontmatter ``name``).
        description:    short, human-readable description (frontmatter).
        instructions:   the Markdown body — what the agent reads when it
            decides to use this skill.
        license:        optional license string.
        allowed_tools:  optional list of tool names the skill expects to
            have available. Advisory — registries don't enforce it.
        root:           on-disk path to the skill directory. ``None``
            for skills built in code (tests).
    """

    name: str
    description: str
    instructions: str
    license: str | None = None
    allowed_tools: tuple[str, ...] = ()
    root: Path | None = None

    @property
    def scripts_dir(self) -> Path | None:
        """``<root>/scripts/`` if the skill has one, else ``None``."""
        if self.root is None:
            return None
        scripts = self.root / "scripts"
        return scripts if scripts.is_dir() else None


def load_skill(skill_dir: Path) -> Skill:
    """Load one skill from a directory containing ``SKILL.md``."""
    skill_md = skill_dir / SKILL_FILE_NAME
    if not skill_md.is_file():
        raise SkillLoadError(f"missing {SKILL_FILE_NAME} in {skill_dir}")
    raw = skill_md.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw, source=str(skill_md))

    name = frontmatter.get("name")
    if not name:
        raise SkillLoadError(f"{skill_md}: frontmatter missing 'name'")
    description = frontmatter.get("description") or ""
    license_str = frontmatter.get("license")

    allowed = frontmatter.get("allowed-tools") or frontmatter.get("allowed_tools")
    if isinstance(allowed, str):
        allowed_list: tuple[str, ...] = tuple(
            x.strip() for x in allowed.split(",") if x.strip()
        )
    elif isinstance(allowed, list):
        allowed_list = tuple(str(x) for x in allowed)
    else:
        allowed_list = ()

    return Skill(
        name=str(name),
        description=str(description),
        instructions=body.strip(),
        license=str(license_str) if license_str else None,
        allowed_tools=allowed_list,
        root=skill_dir,
    )


def load_skills(root: Path | str) -> tuple[Skill, ...]:
    """Discover every ``SKILL.md`` directory under ``root``.

    Skips hidden dirs (``.`` prefix). A ``SKILL.md`` directly in
    ``root`` counts as a single skill (so callers can pass a path to
    one skill or a path to a folder of skills).
    """
    root_path = Path(root)
    if not root_path.exists():
        raise SkillLoadError(f"skills root does not exist: {root_path}")
    if not root_path.is_dir():
        raise SkillLoadError(f"skills root is not a directory: {root_path}")

    candidates: list[Path] = []
    if (root_path / SKILL_FILE_NAME).is_file():
        candidates.append(root_path)
    else:
        for entry in sorted(root_path.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if (entry / SKILL_FILE_NAME).is_file():
                candidates.append(entry)

    return tuple(load_skill(p) for p in candidates)


def _split_frontmatter(
    raw: str, *, source: str,
) -> tuple[dict[str, _FrontmatterValue], str]:
    """Return ``(frontmatter_dict, body)``.

    A SKILL.md without frontmatter is allowed (treated as
    ``frontmatter={}, body=raw``) but loaders that need ``name`` will
    error downstream.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        return {}, raw
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == _FRONTMATTER_DELIM:
            end = i
            break
    if end == -1:
        raise SkillLoadError(
            f"{source}: frontmatter opened with --- but never closed",
        )
    fm_text = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1 :])
    return _parse_simple_yaml(fm_text, source=source), body


_FrontmatterValue = str | list[str]


@dataclass
class _ParseState:
    current_key: str | None = None
    list_buffer: list[str] = field(default_factory=list)


def _parse_simple_yaml(
    text: str, *, source: str,
) -> dict[str, _FrontmatterValue]:
    """Parse a restricted subset of YAML.

    Supported:
        key: value           # scalar string
        key:                 # start a list
          - item1
          - item2
        # comment            # ignored
        empty lines          # ignored

    The minimalism is intentional — Anthropic skill frontmatter is a
    flat list of named fields plus an optional string list, so the
    parser can stay tight without pulling in PyYAML.
    """
    out: dict[str, _FrontmatterValue] = {}
    state = _ParseState()

    def _flush_list() -> None:
        if state.current_key is not None and state.list_buffer:
            out[state.current_key] = list(state.list_buffer)
        state.current_key = None
        state.list_buffer = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        leading = len(line) - len(stripped)

        if stripped.startswith("- ") and state.current_key is not None and leading > 0:
            state.list_buffer.append(_unquote(stripped[2:].strip()))
            continue

        # New top-level key.
        _flush_list()
        if ":" not in stripped:
            raise SkillLoadError(
                f"{source}: cannot parse frontmatter line: {raw_line!r}",
            )
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            raise SkillLoadError(
                f"{source}: empty key in frontmatter line: {raw_line!r}",
            )
        if value == "":
            state.current_key = key
            state.list_buffer = []
        else:
            out[key] = _unquote(value)

    _flush_list()
    return out


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


__all__: Iterable[str] = (
    "Skill",
    "SkillLoadError",
    "load_skill",
    "load_skills",
)
