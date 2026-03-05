"""Start a fresh conversation in the current UI session."""

from __future__ import annotations

from typing import Any

DASH_COMMANDS = [
    {"command": "/new", "description": "Start a new conversation"},
]


def match(raw: str) -> bool:
    return raw.strip().lower() == "/new"


def run(raw: str, *, app: Any) -> bool:
    del raw
    starter = getattr(app, "start_new_conversation", None)
    if callable(starter):
        starter()
        return True
    line = getattr(app, "_append_system_line", None)
    if callable(line):
        line("This UI does not support /new.")
    return True
