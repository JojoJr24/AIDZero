"""Workspace path guards for generated agents."""

from __future__ import annotations

from pathlib import Path


class WorkspaceGuard:
    """Restricts file operations to a single workspace root."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    def resolve(self, relative_path: str) -> Path:
        normalized = _normalize_relative_path(relative_path)
        candidate = (self.workspace_root / normalized).resolve()
        return ensure_within_workspace(candidate, self.workspace_root)

    def mkdir(self, relative_path: str) -> Path:
        target = self.resolve(relative_path)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_text(self, relative_path: str, content: str) -> Path:
        target = self.resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def append_text(self, relative_path: str, content: str) -> Path:
        target = self.resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return target


def ensure_within_workspace(path: Path, workspace_root: Path) -> Path:
    resolved_root = workspace_root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Path escapes workspace root: {path}") from error
    return resolved_path


def resolve_workspace_path(*, workspace_root: Path, relative_path: str) -> Path:
    guard = WorkspaceGuard(workspace_root)
    return guard.resolve(relative_path)


def safe_mkdir(*, workspace_root: Path, relative_path: str) -> Path:
    guard = WorkspaceGuard(workspace_root)
    return guard.mkdir(relative_path)


def safe_write_text(*, workspace_root: Path, relative_path: str, content: str) -> Path:
    guard = WorkspaceGuard(workspace_root)
    return guard.write_text(relative_path, content)


def safe_append_text(*, workspace_root: Path, relative_path: str, content: str) -> Path:
    guard = WorkspaceGuard(workspace_root)
    return guard.append_text(relative_path, content)


def _normalize_relative_path(relative_path: str) -> str:
    raw = (relative_path or "").strip()
    if not raw:
        raise ValueError("Relative path cannot be empty.")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {relative_path}")
    parts = [part for part in candidate.parts if part not in {"", "."}]
    if not parts:
        raise ValueError("Relative path cannot resolve to workspace root.")
    if any(part == ".." for part in parts):
        raise ValueError(f"Parent traversal is not allowed: {relative_path}")
    return Path(*parts).as_posix()
