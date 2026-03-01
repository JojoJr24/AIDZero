"""Dynamic UI discovery and execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


class UIRegistry:
    """Discover runnable UIs in `UI/<name>.py`."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.ui_root = self.repo_root / "UI"

    def names(self) -> list[str]:
        return [path.stem for path in self._iter_ui_files()]

    def run(self, ui_name: str, **kwargs: Any) -> int:
        normalized = ui_name.strip()
        if not normalized:
            raise ValueError("ui_name cannot be empty.")

        ui_module_file = self.ui_root / f"{normalized}.py"
        if not ui_module_file.is_file():
            raise FileNotFoundError(f"UI module not found: {ui_module_file}")

        module_name = f"aidzero_ui_{normalized.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, ui_module_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load UI module: {ui_module_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        run_ui = getattr(module, "run_ui", None)
        if not callable(run_ui):
            raise RuntimeError(f"UI '{normalized}' must export run_ui(...).")

        result = run_ui(**kwargs)
        return int(result) if isinstance(result, int) else 0

    def _iter_ui_files(self) -> list[Path]:
        if not self.ui_root.is_dir():
            return []
        files = [
            path
            for path in self.ui_root.glob("*.py")
            if path.is_file() and path.name != "__init__.py" and not path.name.startswith("_")
        ]
        return sorted(files, key=lambda item: item.name.lower())
