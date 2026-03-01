"""Persist one memory key-value pair."""

from __future__ import annotations

from typing import Any

TOOL_NAME = "memory_set"
TOOL_DESCRIPTION = "Persist a memory key-value pair."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "value": {},
    },
    "required": ["key", "value"],
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del repo_root
    key = str(arguments.get("key", "")).strip()
    if not key:
        raise ValueError("'key' is required.")
    memory.set(key, arguments.get("value"))
    return {"key": key, "status": "stored"}
