"""Write text files under repository scope."""

from __future__ import annotations

from typing import Any

from TOOLS._helpers import safe_resolve

TOOL_NAME = "write_text"
TOOL_DESCRIPTION = "Write UTF-8 text to a repository file."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "append": {"type": "boolean"},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    target = safe_resolve(repo_root, str(arguments.get("path", "")))
    content = str(arguments.get("content", ""))
    append_mode = bool(arguments.get("append", False))

    target.parent.mkdir(parents=True, exist_ok=True)
    if append_mode:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
    else:
        target.write_text(content, encoding="utf-8")

    return {
        "path": str(target.relative_to(repo_root)),
        "bytes_written": len(content.encode("utf-8")),
        "append": append_mode,
    }
