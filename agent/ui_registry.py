"""Dynamic UI discovery and execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


class UIRegistry:
    """Discover runnable UIs in `UI/<name>/entrypoint.py`."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.ui_root = self.repo_root / "UI"

    def names(self) -> list[str]:
        if not self.ui_root.exists():
            return []
        names: list[str] = []
        for entry in sorted(self.ui_root.iterdir(), key=lambda item: item.name.lower()):
            if entry.is_dir() and (entry / "entrypoint.py").is_file():
                names.append(entry.name)
        return names

    def run(self, ui_name: str, **kwargs: Any) -> int:
        normalized = ui_name.strip()
        if not normalized:
            raise ValueError("ui_name cannot be empty.")

        entrypoint = self.ui_root / normalized / "entrypoint.py"
        if not entrypoint.is_file():
            raise FileNotFoundError(f"UI entrypoint not found: {entrypoint}")

        module_name = f"aidzero_ui_{normalized.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, entrypoint)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load UI module: {entrypoint}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        run_ui = getattr(module, "run_ui", None)
        if not callable(run_ui):
            raise RuntimeError(f"UI '{normalized}' must export run_ui(...).")

        result = run_ui(**kwargs)
        return int(result) if isinstance(result, int) else 0
