"""List repository files and directories."""

from __future__ import annotations

from typing import Any

from TOOLS._helpers import safe_resolve

TOOL_NAME = "list_files"
TOOL_DESCRIPTION = "List files and directories under a relative path."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "max_entries": {"type": "integer", "minimum": 1, "maximum": 500},
    },
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    target = safe_resolve(repo_root, str(arguments.get("path", ".")))
    if not target.exists():
        return {"path": str(target.relative_to(repo_root)), "items": [], "exists": False}

    max_entries = int(arguments.get("max_entries", 120))
    items: list[str] = []
    if target.is_file():
        items = [str(target.relative_to(repo_root))]
    else:
        for path in sorted(target.rglob("*")):
            if len(items) >= max_entries:
                break
            items.append(str(path.relative_to(repo_root)))

    return {
        "path": str(target.relative_to(repo_root)),
        "exists": True,
        "items": items,
        "truncated": len(items) >= max_entries,
    }
