"""Describe one MCP tool schema and metadata via gateway."""

from __future__ import annotations

from typing import Any

from TOOLS._mcp_gateway import call_gateway_tool
from TOOLS._helpers import normalize_timeout

TOOL_NAME = "mcp_describe_tool"
TOOL_DESCRIPTION = "Describe one MCP tool by tool_id to inspect schema and risk metadata."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "tool_id": {"type": "string"},
        "force_refresh": {"type": "boolean"},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 180},
    },
    "required": ["tool_id"],
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    tool_id = str(arguments.get("tool_id", "")).strip()
    if not tool_id:
        raise ValueError("'tool_id' is required.")

    payload: dict[str, Any] = {"tool_id": tool_id}
    if "force_refresh" in arguments:
        payload["forceRefresh"] = bool(arguments["force_refresh"])

    timeout = normalize_timeout(arguments, default=60, maximum=180)
    return call_gateway_tool(
        repo_root=repo_root,
        gateway_tool="tool_describe",
        payload=payload,
        timeout_seconds=timeout,
    )
