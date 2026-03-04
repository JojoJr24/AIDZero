"""Trigger selection commands."""

from __future__ import annotations

from typing import Any

_VALID = ("interactive", "heartbeat", "cron", "messengers", "webhooks", "all")

DASH_COMMANDS = [
    {"command": f"/trigger {name}", "description": f"Set trigger to {name}"}
    for name in _VALID
]


def match(raw: str) -> bool:
    return raw.startswith("/trigger ")


def run(raw: str, *, app: Any) -> bool:
    candidate = raw.removeprefix("/trigger ").strip().lower()
    if candidate not in _VALID:
        app._append_system_line("Invalid trigger.")
        return True
    app.active_trigger = candidate
    app._update_status()
    app._append_system_line(f"Active trigger set to: {candidate}")
    return True
