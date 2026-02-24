"""Dynamic registry for runtime UI modules."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UISpec:
    """Single UI registration entry."""

    name: str
    entrypoint: Path
    run_function: str = "run_ui"


class UIRegistry:
    """Loads runnable UI modules from UI/*/entrypoint.py."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self._uis: dict[str, UISpec] = self._discover_ui_specs()

    def names(self) -> list[str]:
        return sorted(self._uis.keys())

    def has(self, ui_name: str) -> bool:
        return ui_name in self._uis

    def run(
        self,
        ui_name: str,
        *,
        provider_name: str,
        model: str,
        user_request: str | None,
        dry_run: bool,
        overwrite: bool,
        yes: bool,
        repo_root: Path,
        ui_options: dict[str, str] | None = None,
    ) -> int:
        spec = self._uis.get(ui_name)
        if spec is None:
            raise ValueError(f"Unknown UI: {ui_name}")

        run_callable = self._load_run_callable(spec)
        result = run_callable(
            provider_name=provider_name,
            model=model,
            user_request=user_request,
            dry_run=dry_run,
            overwrite=overwrite,
            yes=yes,
            repo_root=repo_root,
            ui_options=ui_options or {},
        )
        if isinstance(result, int):
            return result
        raise TypeError(f"UI '{ui_name}' run function must return int exit code.")

    def _discover_ui_specs(self) -> dict[str, UISpec]:
        ui_root = self.repo_root / "UI"
        if not ui_root.is_dir():
            return {}

        specs: dict[str, UISpec] = {}
        for entry in sorted(ui_root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            entrypoint = entry / "entrypoint.py"
            if not entrypoint.exists():
                continue
            specs[entry.name] = UISpec(name=entry.name, entrypoint=entrypoint)
        return specs

    @staticmethod
    def _load_run_callable(spec: UISpec) -> Any:
        module_name = f"aidzero_ui_{spec.name.lower().replace('-', '_')}"
        module_spec = importlib.util.spec_from_file_location(module_name, spec.entrypoint)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Could not load UI module from: {spec.entrypoint}")

        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        run_callable = getattr(module, spec.run_function, None)
        if run_callable is None:
            raise AttributeError(
                f"UI run function '{spec.run_function}' not found in {spec.entrypoint}"
            )
        return run_callable
