"""Simple persistent key-value memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MemoryStore:
    """JSON-backed persistent memory."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, key: str, default: Any = None) -> Any:
        return self._load().get(key, default)

    def set(self, key: str, value: Any) -> None:
        payload = self._load()
        payload[key] = value
        self._save(payload)

    def delete(self, key: str) -> bool:
        payload = self._load()
        if key not in payload:
            return False
        del payload[key]
        self._save(payload)
        return True

    def all(self) -> dict[str, Any]:
        return self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
