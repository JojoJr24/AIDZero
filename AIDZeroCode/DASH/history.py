"""Prompt history command."""

from __future__ import annotations

from typing import Any

DASH_COMMANDS = [
    {"command": "/history", "description": "Show recent prompts"},
]


def match(raw: str) -> bool:
    return raw == "/history"


def run(raw: str, *, app: Any) -> bool:
    del raw
    show_selector = getattr(app, "_show_history_selector", None)
    if callable(show_selector):
        return bool(show_selector(limit=30))

    prompts = app.history.list_prompts(limit=30)
    if not prompts:
        app._append_system_line("History is empty.")
        return True
    app._append_system_line("Recent prompts:")
    for idx, prompt in enumerate(prompts, start=1):
        app._append_system_line(f"{idx:>2}. {prompt}")
    return True
