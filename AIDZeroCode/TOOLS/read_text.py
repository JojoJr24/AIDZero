"""Read text from repository files."""

from __future__ import annotations

from typing import Any

from TOOLS._helpers import safe_resolve

TOOL_NAME = "read_text"
TOOL_DESCRIPTION = "Read UTF-8 text from a repository file."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "max_chars": {"type": "integer", "minimum": 50, "maximum": 200000},
    },
    "required": ["path"],
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    target = safe_resolve(repo_root, str(arguments.get("path", "")))
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {target}")

    max_chars = int(arguments.get("max_chars", 20000))
    text = target.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(target.relative_to(repo_root)),
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
    }
