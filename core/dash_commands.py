"""Dynamic slash-command registry loaded from DASH/*.py."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

Suggestion = tuple[str, str]
DashMatchFn = Callable[[str], bool]
DashRunResult = bool | str | None
DashRunFn = Callable[..., DashRunResult]
LoadedDashCommand = tuple[str, int, list[Suggestion], DashMatchFn, DashRunFn]


@dataclass
class DashCommand:
    """Runtime representation of a slash command module."""

    module: str
    priority: int
    suggestions: list[Suggestion]
    match: DashMatchFn
    run: DashRunFn


class DashCommandRegistry:
    """In-memory slash command catalog."""

    def __init__(self) -> None:
        self._commands: list[DashCommand] = []

    def register(
        self,
        *,
        module: str,
        priority: int,
        suggestions: list[Suggestion],
        match: DashMatchFn,
        run: DashRunFn,
    ) -> None:
        self._commands.append(
            DashCommand(
                module=module,
                priority=priority,
                suggestions=suggestions,
                match=match,
                run=run,
            )
        )
        self._commands.sort(key=lambda item: (item.priority, item.module))

    def is_empty(self) -> bool:
        return not self._commands

    def suggestions(self, query: str) -> list[Suggestion]:
        query_norm = query.strip().lower()
        if not query_norm.startswith("/"):
            return []
        seen: set[str] = set()
        filtered: list[Suggestion] = []
        for command in self._commands:
            for candidate, description in command.suggestions:
                key = candidate.strip().lower()
                if not key or key in seen:
                    continue
                if not key.startswith(query_norm):
                    continue
                seen.add(key)
                filtered.append((candidate.strip(), description.strip()))
        return sorted(filtered, key=lambda item: item[0].lower())

    def is_known_command(self, raw: str) -> bool:
        text = raw.strip()
        return any(command.match(text) for command in self._commands)

    def handle(self, raw: str, *, app: Any) -> bool:
        text = raw.strip()
        for command in self._commands:
            if not command.match(text):
                continue
            return self._normalize_run_result(command.run(text, app=app), app=app)
        return False

    def _normalize_run_result(self, result: DashRunResult, *, app: Any) -> bool:
        if isinstance(result, str):
            setter = getattr(app, "_set_input_text", None)
            if callable(setter):
                setter(result)
            return True
        if result is None:
            return True
        return bool(result)


def build_default_dash_command_registry(
    repo_root: Path,
    *,
    enabled_modules: list[str] | None = None,
) -> DashCommandRegistry:
    """Load slash commands from `DASH/*.py` using a file-based plugin contract."""
    commands_root = repo_root / "DASH"
    registry = DashCommandRegistry()

    enabled_set = {name.strip() for name in enabled_modules or [] if name and name.strip()}

    for command_path in _iter_dash_command_files(commands_root):
        if enabled_set and command_path.stem not in enabled_set:
            continue
        module, priority, suggestions, match, run = _load_dash_command_module(command_path)
        registry.register(
            module=module,
            priority=priority,
            suggestions=suggestions,
            match=match,
            run=run,
        )
    return registry


def _iter_dash_command_files(commands_root: Path) -> list[Path]:
    if not commands_root.is_dir():
        return []
    files = [
        path
        for path in commands_root.glob("*.py")
        if path.is_file() and path.name != "__init__.py" and not path.name.startswith("_")
    ]
    return sorted(files, key=lambda item: item.name.lower())


def _load_dash_command_module(path: Path) -> LoadedDashCommand:
    module_name = f"aidzero_dash_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dash command module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    enabled = bool(getattr(module, "ENABLED", True))
    if not enabled:
        raise RuntimeError(f"Dash command module is disabled: {path.name}")

    raw_suggestions = getattr(module, "DASH_COMMANDS", None)
    match = getattr(module, "match", None)
    run = getattr(module, "run", None)
    raw_priority = getattr(module, "DASH_PRIORITY", 100)

    if not isinstance(raw_suggestions, list) or not raw_suggestions:
        raise RuntimeError(f"Invalid DASH_COMMANDS in {path.name}")
    suggestions = [_normalize_suggestion(item, path.name) for item in raw_suggestions]
    if not callable(match):
        raise RuntimeError(f"Missing callable match(...) in {path.name}")
    if not callable(run):
        raise RuntimeError(f"Missing callable run(...) in {path.name}")
    if not isinstance(raw_priority, int):
        raise RuntimeError(f"Invalid DASH_PRIORITY in {path.name}")

    return path.stem, raw_priority, suggestions, match, run


def _normalize_suggestion(item: Any, filename: str) -> Suggestion:
    if not isinstance(item, dict):
        raise RuntimeError(f"Invalid DASH_COMMANDS item in {filename}")
    command = str(item.get("command", "")).strip()
    description = str(item.get("description", "")).strip()
    if not command.startswith("/"):
        raise RuntimeError(f"Invalid command '{command}' in {filename}")
    if not description:
        raise RuntimeError(f"Missing description for command '{command}' in {filename}")
    return command, description
