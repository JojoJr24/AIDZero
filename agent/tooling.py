"""Tool registry and default tool implementations."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Callable

from agent.memory import MemoryStore
from agent.models import ToolSchema

ToolExecutor = Callable[[dict[str, Any]], Any]


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
        items: list[dict[str, Any]] = []
        for schema in self._schemas.values():
            items.append(
                {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.parameters,
                }
            )
        return sorted(items, key=lambda item: item["name"])

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._executors:
            raise KeyError(f"Unknown tool '{name}'.")
        return self._executors[name](arguments)

    def names(self) -> list[str]:
        return sorted(self._executors.keys())


def build_default_tool_registry(repo_root: Path, memory: MemoryStore) -> ToolRegistry:
    """Create the default tools available to the runtime."""
    registry = ToolRegistry()

    registry.register(
        name="sandbox_run",
        description="Run a shell command inside the repository sandbox.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        execute=lambda args: _tool_sandbox_run(repo_root, args),
    )

    registry.register(
        name="list_files",
        description="List files and directories under a relative path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_entries": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "additionalProperties": False,
        },
        execute=lambda args: _tool_list_files(repo_root, args),
    )

    registry.register(
        name="read_text",
        description="Read UTF-8 text from a repository file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer", "minimum": 50, "maximum": 200000},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        execute=lambda args: _tool_read_text(repo_root, args),
    )

    registry.register(
        name="write_text",
        description="Write UTF-8 text to a repository file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        execute=lambda args: _tool_write_text(repo_root, args),
    )

    registry.register(
        name="memory_set",
        description="Persist a memory key-value pair.",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {},
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
        execute=lambda args: _tool_memory_set(memory, args),
    )

    registry.register(
        name="memory_get",
        description="Read one memory value by key.",
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        execute=lambda args: _tool_memory_get(memory, args),
    )

    registry.register(
        name="memory_list",
        description="Return all persisted memory values.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        execute=lambda _args: memory.all(),
    )

    registry.register(
        name="computer_control",
        description=(
            "Computer-control placeholder. Returns a stub result so the architecture "
            "can wire a real driver later."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "target": {"type": "string"},
                "payload": {},
            },
            "required": ["action"],
            "additionalProperties": True,
        },
        execute=_tool_computer_control,
    )

    return registry


def _safe_resolve(repo_root: Path, raw_path: str | None) -> Path:
    relative = (raw_path or ".").strip() or "."
    resolved = (repo_root / relative).resolve()
    if not str(resolved).startswith(str(repo_root.resolve())):
        raise ValueError("Path escapes repository root.")
    return resolved


def _tool_sandbox_run(repo_root: Path, args: dict[str, Any]) -> dict[str, Any]:
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("'command' is required.")
    timeout = int(args.get("timeout_seconds", 20))
    result = subprocess.run(
        command,
        cwd=repo_root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout[-6000:],
        "stderr": result.stderr[-6000:],
    }


def _tool_list_files(repo_root: Path, args: dict[str, Any]) -> dict[str, Any]:
    target = _safe_resolve(repo_root, str(args.get("path", ".")))
    if not target.exists():
        return {"path": str(target.relative_to(repo_root)), "items": [], "exists": False}

    max_entries = int(args.get("max_entries", 120))
    items: list[str] = []
    if target.is_file():
        items = [str(target.relative_to(repo_root))]
    else:
        for path in sorted(target.rglob("*")):
            if len(items) >= max_entries:
                break
            items.append(str(path.relative_to(repo_root)))
    return {
        "path": str(target.relative_to(repo_root)),
        "exists": True,
        "items": items,
        "truncated": len(items) >= max_entries,
    }


def _tool_read_text(repo_root: Path, args: dict[str, Any]) -> dict[str, Any]:
    target = _safe_resolve(repo_root, str(args.get("path", "")))
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {target}")
    max_chars = int(args.get("max_chars", 20000))
    text = target.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(target.relative_to(repo_root)),
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def _tool_write_text(repo_root: Path, args: dict[str, Any]) -> dict[str, Any]:
    target = _safe_resolve(repo_root, str(args.get("path", "")))
    content = str(args.get("content", ""))
    append_mode = bool(args.get("append", False))
    target.parent.mkdir(parents=True, exist_ok=True)
    if append_mode:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
    else:
        target.write_text(content, encoding="utf-8")
    return {
        "path": str(target.relative_to(repo_root)),
        "bytes_written": len(content.encode("utf-8")),
        "append": append_mode,
    }


def _tool_memory_set(memory: MemoryStore, args: dict[str, Any]) -> dict[str, Any]:
    key = str(args.get("key", "")).strip()
    if not key:
        raise ValueError("'key' is required.")
    memory.set(key, args.get("value"))
    return {"key": key, "status": "stored"}


def _tool_memory_get(memory: MemoryStore, args: dict[str, Any]) -> dict[str, Any]:
    key = str(args.get("key", "")).strip()
    if not key:
        raise ValueError("'key' is required.")
    return {"key": key, "value": memory.get(key)}


def _tool_computer_control(args: dict[str, Any]) -> dict[str, Any]:
    action = str(args.get("action", "")).strip() or "unknown"
    return {
        "status": "not_implemented",
        "action": action,
        "message": (
            "Computer control is wired at architecture level but has no driver yet. "
            "Attach your preferred automation backend in this tool."
        ),
        "request": json.loads(json.dumps(args, ensure_ascii=False)),
    }
