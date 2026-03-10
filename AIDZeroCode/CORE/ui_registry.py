"""Dynamic UI discovery and execution."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Literal
from typing import Any

from CORE.repo_layout import resolve_code_root


UIType = Literal["embedded", "thirdparty"]


class UIRegistry:
    """Discover runnable UIs in `UI/<name>/entrypoint.py`."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.ui_root = resolve_code_root(self.repo_root) / "UI"

    def names(self) -> list[str]:
        return [name for name, _, _ in self._iter_specs()]

    def ui_type(self, ui_name: str) -> UIType:
        normalized = ui_name.strip()
        if not normalized:
            raise ValueError("ui_name cannot be empty.")
        for name, kind, _ in self._iter_specs():
            if name == normalized:
                return kind
        raise FileNotFoundError(f"UI '{normalized}' not found.")

    def run(self, ui_name: str, **kwargs: Any) -> int:
        normalized = ui_name.strip()
        if not normalized:
            raise ValueError("ui_name cannot be empty.")

        ui_kind = self.ui_type(normalized)
        if ui_kind == "thirdparty":
            raise RuntimeError(
                f"UI '{normalized}' is thirdparty and has no local Python entrypoint. "
                "Run only the core API for LAN access."
            )

        ui_entrypoint = self.ui_root / normalized / "entrypoint.py"

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

    def _iter_specs(self) -> list[tuple[str, UIType, Path | None]]:
        if not self.ui_root.is_dir():
            return []
        specs: list[tuple[str, UIType, Path | None]] = []
        for ui_dir in sorted(self.ui_root.iterdir(), key=lambda item: item.name.lower()):
            if not ui_dir.is_dir() or ui_dir.name.startswith("_"):
                continue
            entrypoint = ui_dir / "entrypoint.py"
            if entrypoint.is_file():
                specs.append((ui_dir.name, "embedded", entrypoint))
                continue
            if self._load_type_from_metadata(ui_dir) == "thirdparty":
                specs.append((ui_dir.name, "thirdparty", None))
        return specs

    def _load_type_from_metadata(self, ui_dir: Path) -> UIType | None:
        metadata_file = ui_dir / "ui.json"
        if not metadata_file.is_file():
            return None
        try:
            raw = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        ui_type = str(raw.get("type", "")).strip().lower()
        if ui_type in {"embedded", "thirdparty"}:
            return ui_type
        return None
