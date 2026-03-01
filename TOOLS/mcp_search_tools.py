"""Search MCP tools by intent via gateway."""

from __future__ import annotations

from typing import Any

from TOOLS._mcp_gateway import call_gateway_tool
from TOOLS._helpers import normalize_timeout

TOOL_NAME = "mcp_search_tools"
TOOL_DESCRIPTION = "Search MCP tools by natural-language intent."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        "server": {"type": "string"},
        "force_refresh": {"type": "boolean"},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 180},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("'query' is required.")

    payload: dict[str, Any] = {"query": query}
    if "limit" in arguments:
        payload["limit"] = int(arguments["limit"])
    if "server" in arguments and str(arguments.get("server", "")).strip():
        payload["server"] = str(arguments["server"]).strip()
    if "force_refresh" in arguments:
        payload["forceRefresh"] = bool(arguments["force_refresh"])

    timeout = normalize_timeout(arguments, default=60, maximum=180)
    return call_gateway_tool(
        repo_root=repo_root,
        gateway_tool="tool_search",
        payload=payload,
        timeout_seconds=timeout,
    )
