"""Exit commands."""

from __future__ import annotations

from typing import Any

DASH_COMMANDS = [
    {"command": "/exit", "description": "Exit application"},
    {"command": "/quit", "description": "Exit application"},
]


def match(raw: str) -> bool:
    return raw in {"/exit", "/quit"}


def run(raw: str, *, app: Any) -> bool:
    del raw
    app.exit(0)
    return True
