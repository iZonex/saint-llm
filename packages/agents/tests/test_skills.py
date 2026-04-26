"""Tests for the Anthropic Skills loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from saint_llm_agents import ToolCall, ToolRegistry
from saint_llm_agents.skills import (
    Skill,
    SkillLoadError,
    SkillRegistry,
    load_skill,
    load_skills,
    register_skills_as_tools,
)


def _write_skill(
    root: Path, name: str, *, frontmatter: str, body: str = "Body content.",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")
    return skill_dir


def test_load_skill_parses_basic_frontmatter(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "alpha",
        frontmatter="name: alpha\ndescription: First skill\nlicense: MIT",
        body="# Alpha\n\nDo alpha things.",
    )
    skill = load_skill(skill_dir)
    assert skill.name == "alpha"
    assert skill.description == "First skill"
    assert skill.license == "MIT"
    assert "Do alpha things." in skill.instructions
    assert skill.root == skill_dir


def test_load_skill_parses_allowed_tools_list(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "beta",
        frontmatter=(
            "name: beta\n"
            "description: Second\n"
            "allowed-tools:\n"
            "  - read_file\n"
            "  - run_python\n"
        ),
    )
    skill = load_skill(skill_dir)
    assert skill.allowed_tools == ("read_file", "run_python")


def test_load_skill_parses_allowed_tools_csv_string(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "gamma",
        frontmatter='name: gamma\ndescription: x\nallowed-tools: "a, b, c"',
    )
    skill = load_skill(skill_dir)
    assert skill.allowed_tools == ("a", "b", "c")


def test_load_skill_strips_quotes_from_scalars(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "delta",
        frontmatter='name: "delta"\ndescription: \'A quoted desc\'',
    )
    skill = load_skill(skill_dir)
    assert skill.name == "delta"
    assert skill.description == "A quoted desc"


def test_load_skill_missing_skill_md_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(SkillLoadError, match=r"missing SKILL\.md"):
        load_skill(empty_dir)


def test_load_skill_missing_name_raises(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path, "epsilon", frontmatter="description: nameless",
    )
    with pytest.raises(SkillLoadError, match="missing 'name'"):
        load_skill(skill_dir)


def test_load_skill_unterminated_frontmatter_raises(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bad"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: bad\ndescription: x\n", encoding="utf-8",
    )
    with pytest.raises(SkillLoadError, match="never closed"):
        load_skill(skill_dir)


def test_load_skill_no_frontmatter_treats_whole_file_as_body(tmp_path: Path) -> None:
    skill_dir = tmp_path / "raw"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("Just a body.", encoding="utf-8")
    with pytest.raises(SkillLoadError, match="missing 'name'"):
        load_skill(skill_dir)


def test_load_skills_discovers_subdirs(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", frontmatter="name: alpha\ndescription: a")
    _write_skill(tmp_path, "beta", frontmatter="name: beta\ndescription: b")
    (tmp_path / ".hidden").mkdir()  # should be skipped
    (tmp_path / "no_skill").mkdir()  # no SKILL.md, skipped

    skills = load_skills(tmp_path)
    assert {s.name for s in skills} == {"alpha", "beta"}


def test_load_skills_handles_single_skill_directly(tmp_path: Path) -> None:
    """If root itself is a skill dir, treat it as one skill."""
    skill_dir = _write_skill(tmp_path, "solo", frontmatter="name: solo\ndescription: s")
    skills = load_skills(skill_dir)
    assert len(skills) == 1
    assert skills[0].name == "solo"


def test_load_skills_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(SkillLoadError, match="does not exist"):
        load_skills(tmp_path / "nope")


def test_skill_scripts_dir_returned_when_present(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "withscripts", frontmatter="name: ws\ndescription: x")
    (skill_dir / "scripts").mkdir()
    skill = load_skill(skill_dir)
    assert skill.scripts_dir == skill_dir / "scripts"


def test_skill_scripts_dir_none_when_absent(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "noscripts", frontmatter="name: ns\ndescription: x")
    skill = load_skill(skill_dir)
    assert skill.scripts_dir is None


def test_skill_registry_catalog_format() -> None:
    reg = SkillRegistry([
        Skill(name="a", description="First", instructions="..."),
        Skill(name="b", description="", instructions="..."),
    ])
    cat = reg.catalog(header="Skills:")
    assert "Skills:" in cat
    assert "- a: First" in cat
    assert "- b: (no description)" in cat


def test_skill_registry_catalog_empty_when_no_skills() -> None:
    assert SkillRegistry().catalog() == ""


def test_skill_registry_duplicate_name_raises() -> None:
    reg = SkillRegistry([Skill(name="a", description="x", instructions="...")])
    with pytest.raises(ValueError, match="already registered"):
        reg.add(Skill(name="a", description="y", instructions="..."))


def test_skill_registry_get_unknown_raises() -> None:
    reg = SkillRegistry()
    with pytest.raises(KeyError):
        reg.get("missing")


def test_register_skills_as_tools_exposes_use_prefix() -> None:
    skills = [
        Skill(name="pdf", description="Extract pdf", instructions="step 1: open"),
        Skill(name="csv", description="Parse csv", instructions="step 1: read"),
    ]
    tool_reg = ToolRegistry()
    register_skills_as_tools(tool_reg, skills)

    assert "use_pdf" in tool_reg
    assert "use_csv" in tool_reg
    pdf_msg = tool_reg.execute(ToolCall(id="t", name="use_pdf", arguments={}))
    assert "step 1: open" in pdf_msg.content


def test_register_skills_as_tools_custom_prefix() -> None:
    skills = [Skill(name="x", description="d", instructions="body")]
    tool_reg = ToolRegistry()
    register_skills_as_tools(tool_reg, skills, tool_name_prefix="skill_")
    assert "skill_x" in tool_reg


def test_register_skills_as_tools_accepts_skill_registry() -> None:
    sk_reg = SkillRegistry([
        Skill(name="x", description="d", instructions="body"),
    ])
    tool_reg = ToolRegistry()
    register_skills_as_tools(tool_reg, sk_reg)
    assert "use_x" in tool_reg
