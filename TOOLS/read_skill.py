"""Read one skill definition (SKILL.md + optional references)."""

from __future__ import annotations

from typing import Any

from TOOLS._helpers import safe_resolve

TOOL_NAME = "read_skill"
TOOL_DESCRIPTION = "Read SKILL.md content for one skill, with optional reference files list/content."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "skill_name": {"type": "string"},
        "max_chars": {"type": "integer", "minimum": 100, "maximum": 200000},
        "include_references": {"type": "boolean"},
        "include_reference_contents": {"type": "boolean"},
    },
    "required": ["skill_name"],
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    skill_name = str(arguments.get("skill_name", "")).strip()
    if not skill_name:
        raise ValueError("'skill_name' is required.")

    skill_root = safe_resolve(repo_root, f"SKILLS/{skill_name}")
    skill_md = skill_root / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"Skill not found: {skill_name}")

    max_chars = int(arguments.get("max_chars", 25000))
    content = skill_md.read_text(encoding="utf-8", errors="replace")

    response: dict[str, Any] = {
        "skill_name": skill_name,
        "path": str(skill_root.relative_to(repo_root)),
        "skill_md": content[:max_chars],
        "truncated": len(content) > max_chars,
    }

    if bool(arguments.get("include_references", False)):
        refs_dir = skill_root / "references"
        refs: list[dict[str, Any]] = []
        if refs_dir.is_dir():
            include_contents = bool(arguments.get("include_reference_contents", False))
            remaining = max_chars
            for file in sorted(refs_dir.glob("*")):
                if not file.is_file():
                    continue
                item = {"name": file.name, "path": str(file.relative_to(repo_root))}
                if include_contents:
                    text = file.read_text(encoding="utf-8", errors="replace")
                    chunk = text[:remaining]
                    item["content"] = chunk
                    item["truncated"] = len(text) > len(chunk)
                    remaining = max(0, remaining - len(chunk))
                refs.append(item)
        response["references"] = refs

    return response
