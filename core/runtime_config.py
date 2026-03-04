"""Runtime config persistence for root entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

from core.models import RuntimeConfig


class RuntimeConfigStore:
    """Persists config in `.aidzero/runtime_config.json`."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.path = self.repo_root / ".aidzero" / "runtime_config.json"

    def load(self) -> RuntimeConfig | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None

        ui = str(payload.get("ui", "")).strip()
        provider = str(payload.get("provider", "")).strip()
        model = str(payload.get("model", "")).strip()
        if not ui or not provider or not model:
            return None
        return RuntimeConfig(ui=ui, provider=provider, model=model)

    def save(self, config: RuntimeConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "ui": config.ui,
                    "provider": config.provider,
                    "model": config.model,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
