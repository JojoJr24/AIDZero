"""Run a python script inside a skill's scripts folder."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

from TOOLS._helpers import normalize_timeout, safe_resolve

TOOL_NAME = "run_skill_script"
TOOL_DESCRIPTION = "Execute SKILLS/<skill_name>/scripts/<script>.py with optional argv list."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "skill_name": {"type": "string"},
        "script": {"type": "string"},
        "args": {"type": "array", "items": {"type": "string"}},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
    },
    "required": ["skill_name", "script"],
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    skill_name = str(arguments.get("skill_name", "")).strip()
    script_name = str(arguments.get("script", "")).strip()
    if not skill_name:
        raise ValueError("'skill_name' is required.")
    if not script_name:
        raise ValueError("'script' is required.")

    script_path = safe_resolve(repo_root, f"SKILLS/{skill_name}/scripts/{script_name}")
    if script_path.suffix != ".py":
        raise ValueError("Only .py skill scripts are allowed.")
    if not script_path.is_file():
        raise FileNotFoundError(f"Skill script not found: {script_path}")

    raw_args = arguments.get("args", [])
    if not isinstance(raw_args, list) or any(not isinstance(item, str) for item in raw_args):
        raise ValueError("'args' must be an array of strings.")

    timeout = normalize_timeout(arguments, default=60, maximum=300)
    command = [sys.executable, str(script_path), *raw_args]
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    return {
        "skill_name": skill_name,
        "script": str(Path("SKILLS") / skill_name / "scripts" / script_name),
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout[-6000:],
        "stderr": result.stderr[-6000:],
    }
