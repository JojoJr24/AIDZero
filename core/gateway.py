"""Gateway that transforms external triggers into normalized events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.models import TriggerEvent


class TriggerGateway:
    """Collects heartbeat, cron, messenger, webhook and interactive prompts."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.state_root = self.repo_root / ".aidzero"

    def collect(self, *, trigger: str, prompt: str | None = None, consume: bool = True) -> list[TriggerEvent]:
        mode = trigger.strip().lower() if trigger else "interactive"
        events: list[TriggerEvent] = []
        config = self._load_sources_config()
        heartbeat_path = self._resolve_config_path(config.get("heartbeat_path"), self.repo_root / "HEARTBEAT.md")
        cron_path = self._resolve_config_path(config.get("cron_path"), self.state_root / "cron_prompt.txt")
        message_paths = self._resolve_inbox_paths(
            config.get("message_origins"),
            default=[self.state_root / "inbox" / "messages.jsonl"],
        )
        webhook_paths = self._resolve_inbox_paths(
            config.get("webhook_origins"),
            default=[self.state_root / "inbox" / "webhooks.jsonl"],
        )

        if mode in {"interactive", "all"} and prompt and prompt.strip():
            events.append(
                TriggerEvent(
                    kind="interactive",
                    source="terminal",
                    prompt=prompt.strip(),
                    metadata={"trigger": "interactive"},
                )
            )

        if mode in {"heartbeat", "all"}:
            heartbeat_prompt = self._read_text_file(heartbeat_path)
            if heartbeat_prompt:
                events.append(
                    TriggerEvent(
                        kind="heartbeat",
                        source=self._display_path(heartbeat_path),
                        prompt=heartbeat_prompt,
                        metadata={"trigger": "heartbeat"},
                    )
                )

        if mode in {"cron", "all"}:
            cron_prompt = self._read_text_file(cron_path)
            if cron_prompt:
                events.append(
                    TriggerEvent(
                        kind="cron",
                        source=self._display_path(cron_path),
                        prompt=cron_prompt,
                        metadata={"trigger": "cron"},
                    )
                )

        if mode in {"messengers", "all"}:
            events.extend(self._read_inbox_sources("messages", message_paths, consume=consume))

        if mode in {"webhooks", "all"}:
            events.extend(self._read_inbox_sources("webhooks", webhook_paths, consume=consume))

        return events

    def _read_inbox_sources(self, inbox_name: str, inbox_paths: list[Path], *, consume: bool) -> list[TriggerEvent]:
        events: list[TriggerEvent] = []
        for inbox_path in inbox_paths:
            events.extend(self._read_inbox(inbox_name, inbox_path=inbox_path, consume=consume))
        return events

    def _read_inbox(self, inbox_name: str, *, inbox_path: Path, consume: bool) -> list[TriggerEvent]:
        if not inbox_path.exists():
            return []

        rows = inbox_path.read_text(encoding="utf-8").splitlines()
        events: list[TriggerEvent] = []

        for raw_line in rows:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = {"text": line}

            text = ""
            metadata: dict[str, Any] = {}
            if isinstance(payload, dict):
                if isinstance(payload.get("text"), str):
                    text = payload["text"].strip()
                elif isinstance(payload.get("prompt"), str):
                    text = payload["prompt"].strip()
                metadata = payload
            else:
                text = str(payload).strip()

            if not text:
                continue
            events.append(
                TriggerEvent(
                    kind=inbox_name,
                    source=self._display_path(inbox_path),
                    prompt=text,
                    metadata=metadata,
                )
            )

        if consume:
            inbox_path.unlink(missing_ok=True)
        return events

    @staticmethod
    def _read_text_file(path: Path) -> str:
        if not path.exists() or not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return text

    def _load_sources_config(self) -> dict[str, Any]:
        config_path = self.state_root / "trigger_sources.json"
        if not config_path.exists():
            return {}
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _resolve_config_path(self, raw_path: Any, default: Path) -> Path:
        if isinstance(raw_path, str) and raw_path.strip():
            candidate = Path(raw_path.strip()).expanduser()
            if not candidate.is_absolute():
                candidate = (self.repo_root / candidate).resolve()
            return candidate
        return default

    def _resolve_inbox_paths(self, raw_origins: Any, *, default: list[Path]) -> list[Path]:
        if not isinstance(raw_origins, list) or not raw_origins:
            return default
        resolved: list[Path] = []
        seen: set[Path] = set()
        for item in raw_origins:
            if not isinstance(item, dict):
                continue
            raw_path = item.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            candidate = Path(raw_path.strip()).expanduser()
            if not candidate.is_absolute():
                candidate = (self.repo_root / candidate).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            resolved.append(candidate)
        return resolved or default

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)
