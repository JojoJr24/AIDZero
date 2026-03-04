"""Slash command to load trigger sources from the TUI input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TARGET_ALIASES = {
    "heartbeat": "heartbeat",
    "cron": "cron",
    "message": "messages",
    "messages": "messages",
    "webhook": "webhooks",
    "webhooks": "webhooks",
}

DASH_COMMANDS = [
    {"command": "/load heartbeat <text>", "description": "Set HEARTBEAT.md content"},
    {"command": "/load cron <text>", "description": "Set .aidzero/cron_prompt.txt content"},
    {"command": "/load message <text>", "description": "Append message to inbox"},
    {"command": "/load webhook <text>", "description": "Append webhook to inbox"},
]


def match(raw: str) -> bool:
    return raw.startswith("/load")


def run(raw: str, *, app: Any) -> bool:
    repo_root = getattr(app, "repo_root", None)
    if not isinstance(repo_root, Path):
        app._append_system_line("Cannot resolve repository root for /load.")
        return True

    parsed = _parse_command(raw)
    if parsed is None:
        app._append_system_line("Usage: /load <heartbeat|cron|message|webhook> <text>")
        return True

    target, text = parsed
    if target == "heartbeat":
        heartbeat_path = repo_root / "HEARTBEAT.md"
        heartbeat_path.write_text(text + "\n", encoding="utf-8")
        app._append_system_line("Updated HEARTBEAT.md.")
        return True

    if target == "cron":
        cron_path = repo_root / ".aidzero" / "cron_prompt.txt"
        cron_path.parent.mkdir(parents=True, exist_ok=True)
        cron_path.write_text(text + "\n", encoding="utf-8")
        app._append_system_line("Updated .aidzero/cron_prompt.txt.")
        return True

    inbox_path = repo_root / ".aidzero" / "inbox" / f"{target}.jsonl"
    inbox_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"text": text}
    with inbox_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    app._append_system_line(f"Queued 1 item in {inbox_path.relative_to(repo_root)}.")
    return True


def _parse_command(raw: str) -> tuple[str, str] | None:
    parts = raw.strip().split(maxsplit=2)
    if len(parts) < 3:
        return None
    _, target_raw, text_raw = parts
    target = _TARGET_ALIASES.get(target_raw.strip().lower())
    text = text_raw.strip()
    if not target or not text:
        return None
    return target, text
