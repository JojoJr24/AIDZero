"""List all stored memory values."""

from __future__ import annotations

from typing import Any

TOOL_NAME = "memory_list"
TOOL_DESCRIPTION = "Return all persisted memory values."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del arguments, repo_root
    return memory.all()
