"""Helper functions shared by tool modules."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


def safe_resolve(repo_root: Path, raw_path: str | None) -> Path:
    relative = (raw_path or ".").strip() or "."
    resolved = (repo_root / relative).resolve()
    if not str(resolved).startswith(str(repo_root.resolve())):
        raise ValueError("Path escapes repository root.")
    return resolved


def normalize_timeout(arguments: dict[str, Any], *, default: int, maximum: int) -> int:
    timeout = int(arguments.get("timeout_seconds", default))
    return max(1, min(timeout, maximum))


def run_command(command: list[str], *, timeout_seconds: int = 20) -> dict[str, Any]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout[-6000:],
        "stderr": result.stderr[-6000:],
    }


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise RuntimeError(f"Required binary not found: {name}")
    return binary


def require_desktop_session() -> None:
    if sys.platform.startswith("linux"):
        if not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
            raise RuntimeError("No GUI session detected (DISPLAY/WAYLAND_DISPLAY missing).")
