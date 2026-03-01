"""Read runtime history entries from `.aidzero/store/history.jsonl`."""

from __future__ import annotations

import json
from typing import Any

from TOOLS._helpers import safe_resolve

TOOL_NAME = "history_get"
TOOL_DESCRIPTION = "Read recent runtime history entries, optionally filtered by a query."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        "query": {"type": "string"},
    },
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    history_path = safe_resolve(repo_root, ".aidzero/store/history.jsonl")
    if not history_path.exists():
        return {"entries": [], "count": 0, "source": str(history_path.relative_to(repo_root))}

    limit = int(arguments.get("limit", 25))
    query = str(arguments.get("query", "")).strip().lower()

    rows: list[dict[str, Any]] = []
    for line in history_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)

    if query:
        filtered: list[dict[str, Any]] = []
        for row in rows:
            haystack = json.dumps(row, ensure_ascii=False).lower()
            if query in haystack:
                filtered.append(row)
        rows = filtered

    entries = rows[-limit:]
    return {
        "entries": entries,
        "count": len(entries),
        "source": str(history_path.relative_to(repo_root)),
    }
