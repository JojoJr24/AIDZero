"""Prompt history persistence for terminal/web style UIs."""

from __future__ import annotations

import json
from pathlib import Path


class PromptHistoryStore:
    """Stores prompts in `.aidzero/prompt_history.json`."""

    def __init__(self, repo_root: Path, *, max_items: int = 200) -> None:
        self.repo_root = repo_root.resolve()
        self.max_items = max(1, max_items)
        self.filepath = self.repo_root / ".aidzero" / "prompt_history.json"

    def list_prompts(self, *, limit: int | None = None) -> list[str]:
        items = self._load()
        if limit is None or limit <= 0:
            return items
        return items[:limit]

    def add_prompt(self, prompt: str) -> list[str]:
        text = prompt.strip()
        if not text:
            return self.list_prompts()

        items = [item for item in self._load() if item != text]
        items.insert(0, text)
        items = items[: self.max_items]
        self._save(items)
        return items

    def _load(self) -> list[str]:
        if not self.filepath.exists():
            return []
        try:
            payload = json.loads(self.filepath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, dict):
            return []
        raw_items = payload.get("prompts")
        if not isinstance(raw_items, list):
            return []

        unique: list[str] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, str):
                continue
            item = raw_item.strip()
            if item and item not in unique:
                unique.append(item)
        return unique[: self.max_items]

    def _save(self, items: list[str]) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {"prompts": items}
        self.filepath.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
