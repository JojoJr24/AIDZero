"""Read one memory value by key."""

from __future__ import annotations

from typing import Any

TOOL_NAME = "memory_get"
TOOL_DESCRIPTION = "Read one memory value by key."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
    },
    "required": ["key"],
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del repo_root
    key = str(arguments.get("key", "")).strip()
    if not key:
        raise ValueError("'key' is required.")
    return {"key": key, "value": memory.get(key)}
