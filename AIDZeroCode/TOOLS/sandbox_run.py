"""Run shell commands in repository scope."""

from __future__ import annotations

import subprocess
from typing import Any

TOOL_NAME = "sandbox_run"
TOOL_DESCRIPTION = "Run a shell command inside the repository sandbox."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
    },
    "required": ["command"],
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    command = str(arguments.get("command", "")).strip()
    if not command:
        raise ValueError("'command' is required.")

    timeout = int(arguments.get("timeout_seconds", 20))
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
