"""Shared helper for MCP gateway client tools."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from TOOLS._helpers import require_binary
from CORE.repo_layout import resolve_code_root


def call_gateway_tool(
    *,
    repo_root: Path,
    gateway_tool: str,
    payload: dict[str, Any] | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.is_file():
        python_bin = str(venv_python)
    else:
        python_bin = require_binary("python3")
    code_root = resolve_code_root(repo_root)
    script_path = code_root / "MCP" / "tool-gateway" / "scripts" / "gateway-call.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Gateway caller script not found: {script_path}")

    raw_payload = json.dumps(payload or {}, ensure_ascii=False)
    command = [
        python_bin,
        str(script_path),
        "--tool",
        gateway_tool,
        "--payload",
        raw_payload,
    ]

    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_seconds),
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"Gateway call timed out (tool={gateway_tool}, timeout_seconds={timeout_seconds})."
        ) from error

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        raise RuntimeError(
            "Gateway call failed "
            f"(tool={gateway_tool}, exit_code={result.returncode}). "
            f"stderr={stderr[-3000:]!r} stdout={stdout[-3000:]!r}"
        )

    output_text = (result.stdout or "").strip()
    if not output_text:
        raise RuntimeError(f"Gateway returned empty stdout for tool '{gateway_tool}'.")

    try:
        response = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Gateway returned non-JSON output for tool '{gateway_tool}': {output_text[:1000]}"
        ) from error

    if not isinstance(response, dict):
        raise RuntimeError(f"Gateway output is not an object for tool '{gateway_tool}'.")

    return response
