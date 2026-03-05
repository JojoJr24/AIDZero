"""Helpers to resolve project/code roots across launcher layouts."""

from __future__ import annotations

from pathlib import Path


def resolve_code_root(repo_root: Path) -> Path:
    """Return directory containing runtime code folders (UI/TOOLS/LLMProviders/MCP)."""

    root = repo_root.resolve()
    nested = (root / "AIDZeroCode").resolve()
    return nested if nested.is_dir() else root

