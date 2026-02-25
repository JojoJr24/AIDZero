"""Persistent prompt history shared by UI runtimes."""

from __future__ import annotations

import json
from pathlib import Path

HISTORY_DIRNAME = ".aidzero"
HISTORY_FILENAME = "prompt_history.json"
DEFAULT_MAX_PROMPTS = 50


class PromptHistoryStore:
    """Stores prompt history in a repository-local JSON file."""

    def __init__(self, repo_root: Path, *, max_prompts: int = DEFAULT_MAX_PROMPTS) -> None:
        self.repo_root = repo_root.resolve()
        self.max_prompts = max(1, int(max_prompts))
        self.path = self.repo_root / HISTORY_DIRNAME / HISTORY_FILENAME

    def list_prompts(self, *, limit: int | None = None) -> list[str]:
        prompts = self._load_prompts()
        if limit is None:
            return prompts
        if limit <= 0:
            return []
        return prompts[:limit]

    def add_prompt(self, prompt: str) -> list[str]:
        normalized = prompt.strip()
        if not normalized:
            return self.list_prompts()

        prompts = [item for item in self._load_prompts() if item != normalized]
        prompts.insert(0, normalized)
        prompts = prompts[: self.max_prompts]
        self._save_prompts(prompts)
        return prompts

    def _load_prompts(self) -> list[str]:
        if not self.path.exists():
            return []

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []

        if isinstance(payload, list):
            source_items = payload
        elif isinstance(payload, dict):
            source_items = payload.get("prompts", [])
        else:
            return []

        prompts: list[str] = []
        for item in source_items:
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            if stripped:
                prompts.append(stripped)
        return prompts[: self.max_prompts]

    def _save_prompts(self, prompts: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"prompts": prompts}
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
