"""Gateway that transforms triggers into normalized events."""

from __future__ import annotations

from pathlib import Path

from core.models import TriggerEvent


class TriggerGateway:
    """Collects interactive prompts."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def collect(self, *, trigger: str, prompt: str | None = None, consume: bool = True) -> list[TriggerEvent]:
        del consume
        mode = trigger.strip().lower() if trigger else "interactive"
        if mode != "interactive":
            return []
        if not prompt or not prompt.strip():
            return []
        return [
            TriggerEvent(
                kind="interactive",
                source="terminal",
                prompt=prompt.strip(),
                metadata={"trigger": "interactive"},
            )
        ]
