"""Dynamic tool registry loaded from TOOLS/*.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

from core.memory import MemoryStore
from core.models import ToolSchema

ToolExecutor = Callable[[dict[str, Any]], Any]
LoadedTool = tuple[str, str, dict[str, Any], Callable[..., Any]]


class ToolRegistry:
    """In-memory tool catalog with JSON schemas."""

    def __init__(self) -> None:
        self._schemas: dict[str, ToolSchema] = {}
        self._executors: dict[str, ToolExecutor] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        execute: ToolExecutor,
    ) -> None:
        tool_name = name.strip()
        if not tool_name:
            raise ValueError("Tool name cannot be empty.")
        self._schemas[tool_name] = ToolSchema(
            name=tool_name,
            description=description.strip(),
            parameters=parameters,
        )
        self._executors[tool_name] = execute

    def schemas(self) -> list[dict[str, Any]]:
        return sorted(
            [
                {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.parameters,
                }
                for schema in self._schemas.values()
            ],
            key=lambda item: item["name"],
        )

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._executors:
            raise KeyError(f"Unknown tool '{name}'.")
        return self._executors[name](arguments)

    def names(self) -> list[str]:
        return sorted(self._executors.keys())


def build_default_tool_registry(
    repo_root: Path,
    memory: MemoryStore,
    *,
    enabled_names: list[str] | None = None,
    disabled_names: list[str] | None = None,
) -> ToolRegistry:
    """Load tools from `TOOLS/*.py` using a file-based plugin contract."""
    tools_root = repo_root / "TOOLS"
    registry = ToolRegistry()

    enabled_set = {name.strip() for name in enabled_names or [] if name and name.strip()}
    disabled_set = {name.strip() for name in disabled_names or [] if name and name.strip()}

    for tool_path in _iter_tool_files(tools_root):
        name, description, parameters, runner = _load_tool_module(tool_path)
        if name in disabled_set:
            continue
        if enabled_set and name not in enabled_set:
            continue
        registry.register(
            name=name,
            description=description,
            parameters=parameters,
            execute=lambda arguments, fn=runner: fn(arguments, repo_root=repo_root, memory=memory),
        )

    if not registry.names():
        raise RuntimeError(
            f"No tools loaded from {tools_root}. Add at least one valid tool module in TOOLS/*.py."
        )
    return registry


def _iter_tool_files(tools_root: Path) -> list[Path]:
    if not tools_root.is_dir():
        return []
    files = [
        path
        for path in tools_root.glob("*.py")
        if path.is_file() and path.name != "__init__.py" and not path.name.startswith("_")
    ]
    return sorted(files, key=lambda item: item.name.lower())


def _load_tool_module(path: Path) -> LoadedTool:
    module_name = f"aidzero_tool_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load tool module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    enabled = bool(getattr(module, "ENABLED", True))
    if not enabled:
        raise RuntimeError(f"Tool module is disabled: {path.name}")

    name = getattr(module, "TOOL_NAME", None)
    description = getattr(module, "TOOL_DESCRIPTION", None)
    parameters = getattr(module, "TOOL_PARAMETERS", None)
    run = getattr(module, "run", None)

    if not isinstance(name, str) or not name.strip():
        raise RuntimeError(f"Invalid TOOL_NAME in {path.name}")
    if not isinstance(description, str):
        raise RuntimeError(f"Invalid TOOL_DESCRIPTION in {path.name}")
    if not isinstance(parameters, dict):
        raise RuntimeError(f"Invalid TOOL_PARAMETERS in {path.name}")
    if not callable(run):
        raise RuntimeError(f"Missing callable run(...) in {path.name}")

    return name.strip(), description.strip(), parameters, run
