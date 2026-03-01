"""List/search MCP tools exposed by the tool gateway."""

from __future__ import annotations

from typing import Any

from TOOLS._mcp_gateway import call_gateway_tool
from TOOLS._helpers import normalize_timeout

TOOL_NAME = "mcp_search_tools"
TOOL_DESCRIPTION = "List MCP tools from the tool gateway."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        "server": {"type": "string"},
        "force_refresh": {"type": "boolean"},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 180},
    },
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    query = "list available tools"

    payload: dict[str, Any] = {"query": query}
    if "limit" in arguments:
        payload["limit"] = int(arguments["limit"])
    if "server" in arguments and str(arguments.get("server", "")).strip():
        payload["server"] = str(arguments["server"]).strip()
    if "force_refresh" in arguments:
        payload["forceRefresh"] = bool(arguments["force_refresh"])

    timeout = normalize_timeout(arguments, default=60, maximum=180)
    raw = call_gateway_tool(
        repo_root=repo_root,
        gateway_tool="tool_search",
        payload=payload,
        timeout_seconds=timeout,
    )
    matches = raw.get("matches") if isinstance(raw, dict) else None
    if not isinstance(matches, list):
        # Some gateways wrap results in structuredContent.
        structured = raw.get("structuredContent") if isinstance(raw, dict) else None
        matches = structured.get("matches") if isinstance(structured, dict) else None
    normalized_tools: list[dict[str, Any]] = []
    if isinstance(matches, list):
        for item in matches:
            if not isinstance(item, dict):
                continue
            tool_id = item.get("id")
            server_name = item.get("server")
            tool_name = item.get("tool")
            if not isinstance(tool_id, str) or not tool_id.strip():
                continue
            normalized_tools.append(
                {
                    "tool_id": tool_id.strip(),
                    "server": server_name if isinstance(server_name, str) else "",
                    "tool": tool_name if isinstance(tool_name, str) else "",
                    "description": item.get("description", ""),
                    "risk": item.get("risk", ""),
                }
            )

    return {
        "tools": normalized_tools,
        "count": len(normalized_tools),
        "query_used": query,
        "raw": raw,
    }
