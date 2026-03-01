"""Check MCP gateway health/state."""

from __future__ import annotations

from typing import Any

from TOOLS._mcp_gateway import call_gateway_tool
from TOOLS._helpers import normalize_timeout

TOOL_NAME = "mcp_health"
TOOL_DESCRIPTION = "Check MCP gateway health and indexed-tool statistics."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 180},
    },
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    timeout = normalize_timeout(arguments, default=45, maximum=180)
    return call_gateway_tool(
        repo_root=repo_root,
        gateway_tool="tool_health",
        payload={},
        timeout_seconds=timeout,
    )
