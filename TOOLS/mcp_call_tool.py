"""Call one MCP tool through gateway by tool_id and args."""

from __future__ import annotations

from typing import Any

from TOOLS._mcp_gateway import call_gateway_tool
from TOOLS._helpers import normalize_timeout

TOOL_NAME = "mcp_call_tool"
TOOL_DESCRIPTION = "Execute one MCP tool by tool_id and JSON args through the gateway."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "tool_id": {"type": "string"},
        "args": {"type": "object"},
        "force_refresh": {"type": "boolean"},
        "timeout_ms": {"type": "integer", "minimum": 1, "maximum": 600000},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
    },
    "required": ["tool_id"],
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    tool_id = str(arguments.get("tool_id", "")).strip()
    if not tool_id:
        raise ValueError("'tool_id' is required.")

    raw_args = arguments.get("args", {})
    if not isinstance(raw_args, dict):
        raise ValueError("'args' must be a JSON object.")

    payload: dict[str, Any] = {
        "tool_id": tool_id,
        "args": raw_args,
    }
    if "force_refresh" in arguments:
        payload["forceRefresh"] = bool(arguments["force_refresh"])
    if "timeout_ms" in arguments:
        payload["timeoutMs"] = int(arguments["timeout_ms"])

    timeout = normalize_timeout(arguments, default=90, maximum=300)
    return call_gateway_tool(
        repo_root=repo_root,
        gateway_tool="tool_call",
        payload=payload,
        timeout_seconds=timeout,
    )
