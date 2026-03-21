"""Unit tests for the skill loading and matching system."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.skills import (
    Skill,
    _parse_frontmatter,
    _strip_frontmatter,
    format_skills_context,
    load_skills,
    match_skills,
)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_extracts_name_and_description(self) -> None:
        text = (
            "---\n"
            "name: my-skill\n"
            "description: Useful for physics\n"
            "---\n"
            "# Body\n"
        )
        fm = _parse_frontmatter(text)
        assert fm["name"] == "my-skill"
        assert fm["description"] == "Useful for physics"

    def test_strips_quotes(self) -> None:
        text = '---\nversion: "1.0"\n---\n# Body\n'
        fm = _parse_frontmatter(text)
        assert fm["version"] == "1.0"

    def test_returns_empty_when_no_frontmatter(self) -> None:
        assert _parse_frontmatter("# No frontmatter here\n") == {}

    def test_ignores_comment_lines(self) -> None:
        text = "---\nname: test\n# this is a comment\n---\n"
        fm = _parse_frontmatter(text)
        assert fm == {"name": "test"}


class TestStripFrontmatter:
    def test_removes_frontmatter(self) -> None:
        text = "---\nname: x\n---\n\n# Body\nHello"
        assert _strip_frontmatter(text) == "# Body\nHello"

    def test_noop_without_frontmatter(self) -> None:
        text = "# Body\nHello"
        assert _strip_frontmatter(text) == text


# ---------------------------------------------------------------------------
# Skill loading
# ---------------------------------------------------------------------------


class TestLoadSkills:
    def test_loads_skills_from_parent_dir(self, tmp_path: Path) -> None:
        """Standard layout: skills_dir/my-skill/SKILL.md"""
        skill_dir = tmp_path / "alpha"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: Alpha skill\n---\n# Alpha\nDo alpha things.\n"
        )

        skills = load_skills([str(tmp_path)])
        assert len(skills) == 1
        assert skills[0].name == "alpha"
        assert skills[0].description == "Alpha skill"

    def test_loads_single_skill_directory(self, tmp_path: Path) -> None:
        """Direct layout: skills_dir is the skill itself."""
        (tmp_path / "SKILL.md").write_text(
            "---\nname: direct\ndescription: Direct skill\n---\n# Direct\n"
        )
        skills = load_skills([str(tmp_path)])
        assert len(skills) == 1
        assert skills[0].name == "direct"

    def test_last_wins_precedence(self, tmp_path: Path) -> None:
        """When two sources define the same skill name, the last one wins."""
        dir_a = tmp_path / "a" / "dup"
        dir_b = tmp_path / "b" / "dup"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)
        (dir_a / "SKILL.md").write_text(
            "---\nname: dup\ndescription: First\n---\n# First\n"
        )
        (dir_b / "SKILL.md").write_text(
            "---\nname: dup\ndescription: Second\n---\n# Second\n"
        )
        skills = load_skills([str(tmp_path / "a"), str(tmp_path / "b")])
        assert len(skills) == 1
        assert skills[0].description == "Second"

    def test_skips_nonexistent_dir(self) -> None:
        skills = load_skills(["/nonexistent/path/xyz"])
        assert skills == []

    def test_falls_back_to_directory_name(self, tmp_path: Path) -> None:
        """When frontmatter has no name, use the directory name."""
        skill_dir = tmp_path / "my-cool-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: Cool stuff\n---\n# Cool\n"
        )
        skills = load_skills([str(tmp_path)])
        assert skills[0].name == "my-cool-skill"


# ---------------------------------------------------------------------------
# Skill matching
# ---------------------------------------------------------------------------


class TestMatchSkills:
    @pytest.fixture()
    def sample_skills(self, tmp_path: Path) -> list[Skill]:
        s1 = Skill(
            name="quantum",
            description="Quantum mechanics, Schrödinger equation, wavefunctions",
            path=tmp_path / "a" / "SKILL.md",
        )
        s2 = Skill(
            name="optimization",
            description="Numerical optimization, root finding, minimization",
            path=tmp_path / "b" / "SKILL.md",
        )
        return [s1, s2]

    def test_matches_relevant_skill(self, sample_skills: list[Skill]) -> None:
        matched = match_skills(sample_skills, "Solve the Schrödinger equation for a potential well")
        names = [s.name for s in matched]
        assert "quantum" in names

    def test_no_match_returns_empty(self, sample_skills: list[Skill]) -> None:
        matched = match_skills(sample_skills, "Build a REST API for user management")
        assert matched == []

    def test_matches_multiple_skills(self, sample_skills: list[Skill]) -> None:
        matched = match_skills(
            sample_skills,
            "Use numerical optimization to find eigenvalues of the Schrödinger equation",
        )
        names = [s.name for s in matched]
        assert "quantum" in names
        assert "optimization" in names


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormatSkillsContext:
    def test_empty_list_returns_empty_string(self) -> None:
        assert format_skills_context([]) == ""

    def test_includes_skill_name_and_body(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("---\nname: test\n---\n# Test Skill\nDo something.\n")

        skill = Skill(name="test", description="A test", path=skill_file)
        result = format_skills_context([skill])
        assert "### Skill: test" in result
        assert "# Test Skill" in result
        assert "Do something." in result
