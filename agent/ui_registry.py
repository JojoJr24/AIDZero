"""Dynamic UI discovery and execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


class UIRegistry:
    """Discover runnable UIs from `UI/<name>/entrypoint.py`."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.ui_root = self.repo_root / "UI"

    def names(self) -> list[str]:
        items: list[str] = []
        if not self.ui_root.is_dir():
            return items
        for entry in sorted(self.ui_root.iterdir(), key=lambda item: item.name.lower()):
            if entry.is_dir() and (entry / "entrypoint.py").is_file():
                items.append(entry.name)
        return items

    def run(self, ui_name: str, **kwargs: Any) -> int:
        normalized = ui_name.strip()
        if not normalized:
            raise ValueError("ui_name cannot be empty.")
        entrypoint_file = self.ui_root / normalized / "entrypoint.py"
        if not entrypoint_file.is_file():
            raise FileNotFoundError(f"UI entrypoint not found: {entrypoint_file}")

        module_name = f"aidzero_ui_{normalized.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, entrypoint_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load UI entrypoint: {entrypoint_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        run_ui = getattr(module, "run_ui", None)
        if not callable(run_ui):
            raise RuntimeError(f"UI '{normalized}' does not expose run_ui(...).")
        result = run_ui(**kwargs)
        if isinstance(result, int):
            return result
        return 0
