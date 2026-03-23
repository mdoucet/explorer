"""Skill loading and matching for the Scientific Loop agent.

Skills follow the `Agent Skills <https://agentskills.io/specification>`_
pattern: each skill lives in its own directory and contains a ``SKILL.md``
file with YAML frontmatter (``name``, ``description``) followed by
domain-specific instructions.

The loader uses **progressive disclosure** — only the frontmatter is read
during indexing.  Full skill content is loaded only when a skill matches
the current task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    """A single loaded skill.

    Attributes
    ----------
    name : str
        Short identifier (from frontmatter or directory name).
    description : str
        One-line description used for matching.
    path : Path
        Path to the ``SKILL.md`` file.
    metadata : dict[str, Any]
        Any extra frontmatter fields (author, version, …).
    """

    name: str
    description: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    # Lazily cached full content
    _content: str | None = field(default=None, repr=False, compare=False)

    @property
    def content(self) -> str:
        """Full Markdown body (everything after the frontmatter)."""
        if self._content is None:
            raw = self.path.read_text()
            self._content = _strip_frontmatter(raw)
        return self._content


# ------------------------------------------------------------------
# Frontmatter parsing (lightweight, no PyYAML dependency required)
# ------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract simple ``key: value`` pairs from YAML frontmatter.

    Only scalar string values are supported — this avoids pulling in a
    full YAML parser for a handful of metadata fields.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    pairs: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            pairs[key.strip()] = value.strip().strip('"').strip("'")
    return pairs


def _strip_frontmatter(text: str) -> str:
    """Return *text* with the YAML frontmatter block removed."""
    return _FRONTMATTER_RE.sub("", text).strip()


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------


def load_skills(skill_dirs: list[str]) -> list[Skill]:
    """Scan *skill_dirs* for skill folders and build a skill index.

    Each directory in *skill_dirs* is expected to contain sub-directories,
    each with a ``SKILL.md`` file.  *skill_dirs* itself may also be a
    single skill directory (containing ``SKILL.md`` directly).

    Parameters
    ----------
    skill_dirs : list[str]
        Paths to directories containing skill folders.

    Returns
    -------
    list[Skill]
        Indexed skills with metadata parsed from frontmatter.
    """
    skills: list[Skill] = []
    seen_names: set[str] = set()

    for dir_str in skill_dirs:
        root = Path(dir_str).resolve()
        if not root.is_dir():
            continue

        # Check if this is a single skill directory (has SKILL.md directly)
        direct = root / "SKILL.md"
        if direct.is_file():
            skill = _load_one(direct)
            if skill.name not in seen_names:
                seen_names.add(skill.name)
                skills.append(skill)
            continue

        # Otherwise scan sub-directories
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            skill_file = child / "SKILL.md"
            if skill_file.is_file():
                skill = _load_one(skill_file)
                # Last-wins semantics (match Deep Agents precedence)
                if skill.name in seen_names:
                    skills = [s for s in skills if s.name != skill.name]
                seen_names.add(skill.name)
                skills.append(skill)

    return skills


def _load_one(skill_file: Path) -> Skill:
    """Parse a single ``SKILL.md`` into a :class:`Skill`."""
    raw = skill_file.read_text()
    fm = _parse_frontmatter(raw)
    name = fm.pop("name", skill_file.parent.name)
    description = fm.pop("description", "")
    return Skill(
        name=name,
        description=description,
        path=skill_file,
        metadata=fm,
    )


# ------------------------------------------------------------------
# Matching
# ------------------------------------------------------------------


def match_skills(skills: list[Skill], task: str) -> list[Skill]:
    """Return skills whose description is relevant to *task*.

    Matching is intentionally simple: a skill matches when at least one
    substantive word (≥4 chars) from its description appears in the task
    text (case-insensitive).  This keeps the mechanism transparent and
    predictable for scientists who are new to the workflow.
    """
    task_lower = task.lower()
    matched: list[Skill] = []
    for skill in skills:
        words = set(skill.description.lower().split())
        substantive = {w for w in words if len(w) >= 4}
        if any(w in task_lower for w in substantive):
            matched.append(skill)
    return matched


_RECIPE_RE = re.compile(r"^#{2}\s+Recipe\b", re.MULTILINE | re.IGNORECASE)


def format_skills_context(skills: list[Skill]) -> str:
    """Render matched skills into a prompt section.

    Returns an empty string if *skills* is empty.  When a skill contains
    recipe sections (``## Recipe:``), a MUST-USE directive is prepended
    so the LLM treats them as prescriptive rather than advisory.
    """
    if not skills:
        return ""
    sections: list[str] = ["## Relevant Skills\n"]
    for skill in skills:
        has_recipe = bool(_RECIPE_RE.search(skill.content))
        sections.append(f"### Skill: {skill.name}\n")
        if has_recipe:
            sections.append(
                "**⚠️ MANDATORY:** This skill contains a recipe that "
                "directly applies to this task.  You MUST follow the "
                "recipe's algorithm and code pattern.  Do NOT substitute "
                "a different approach (e.g. matrix diagonalisation instead "
                "of root-finding) unless the recipe explicitly says "
                "alternatives are acceptable.\n"
            )
        sections.append(skill.content)
        sections.append("")
    return "\n".join(sections)
