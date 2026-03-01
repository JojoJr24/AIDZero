"""List installed skills available in SKILLS/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

TOOL_NAME = "list_skills"
TOOL_DESCRIPTION = "List available skills from SKILLS/ with descriptions and scripts."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del arguments, memory
    skills_root = repo_root / "SKILLS"
    if not skills_root.is_dir():
        return {"skills": []}

    skills: list[dict[str, Any]] = []
    for entry in sorted(skills_root.iterdir(), key=lambda item: item.name.lower()):
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.is_file():
            continue

        scripts_dir = entry / "scripts"
        scripts = []
        if scripts_dir.is_dir():
            scripts = sorted(
                [
                    file.name
                    for file in scripts_dir.iterdir()
                    if file.is_file() and file.suffix == ".py" and not file.name.startswith("_")
                ]
            )

        skills.append(
            {
                "name": entry.name,
                "path": str(entry.relative_to(repo_root)),
                "description": _extract_description(skill_md),
                "scripts": scripts,
            }
        )
    return {"skills": skills}


def _extract_description(skill_md: Path) -> str | None:
    try:
        lines = skill_md.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    in_frontmatter = False
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter and stripped.startswith("description:"):
            value = stripped.split(":", 1)[1].strip()
            return value.strip('"').strip("'")
    return None
