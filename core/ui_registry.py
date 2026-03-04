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
        return [path.parent.name for path in self._iter_entrypoints()]

    def run(self, ui_name: str, **kwargs: Any) -> int:
        normalized = ui_name.strip()
        if not normalized:
            raise ValueError("ui_name cannot be empty.")

        ui_entrypoint = self.ui_root / normalized / "entrypoint.py"
        if not ui_entrypoint.is_file():
            raise FileNotFoundError(f"UI entrypoint not found: {ui_entrypoint}")

        module_name = f"aidzero_ui_{normalized.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, ui_entrypoint)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load UI module: {ui_entrypoint}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        run_ui = getattr(module, "run_ui", None)
        if not callable(run_ui):
            raise RuntimeError(f"UI '{normalized}' must export run_ui(...).")

        result = run_ui(**kwargs)
        return int(result) if isinstance(result, int) else 0

    def _iter_entrypoints(self) -> list[Path]:
        if not self.ui_root.is_dir():
            return []
        entrypoints: list[Path] = []
        for ui_dir in sorted(self.ui_root.iterdir(), key=lambda item: item.name.lower()):
            if not ui_dir.is_dir() or ui_dir.name.startswith("_"):
                continue
            entrypoint = ui_dir / "entrypoint.py"
            if entrypoint.is_file():
                entrypoints.append(entrypoint)
        return entrypoints
