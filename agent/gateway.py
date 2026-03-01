"""Gateway that transforms external triggers into normalized events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.models import TriggerEvent


class TriggerGateway:
    """Collects heartbeat, cron, messenger, webhook and interactive prompts."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.state_root = self.repo_root / ".aidzero"

    def collect(self, *, trigger: str, prompt: str | None = None, consume: bool = True) -> list[TriggerEvent]:
        mode = trigger.strip().lower() if trigger else "interactive"
        events: list[TriggerEvent] = []

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
            heartbeat_prompt = self._read_text_file(self.repo_root / "HEARTBEAT.md")
            if heartbeat_prompt:
                events.append(
                    TriggerEvent(
                        kind="heartbeat",
                        source="heartbeat-file",
                        prompt=heartbeat_prompt,
                        metadata={"trigger": "heartbeat"},
                    )
                )

        if mode in {"cron", "all"}:
            cron_prompt = self._read_text_file(self.state_root / "cron_prompt.txt")
            if cron_prompt:
                events.append(
                    TriggerEvent(
                        kind="cron",
                        source="cron_prompt.txt",
                        prompt=cron_prompt,
                        metadata={"trigger": "cron"},
                    )
                )

        if mode in {"messengers", "all"}:
            events.extend(self._read_inbox("messages", consume=consume))

        if mode in {"webhooks", "all"}:
            events.extend(self._read_inbox("webhooks", consume=consume))

        return events

    def _read_inbox(self, inbox_name: str, *, consume: bool) -> list[TriggerEvent]:
        inbox_path = self.state_root / "inbox" / f"{inbox_name}.jsonl"
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
                    source=f"inbox:{inbox_name}",
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
