"""List/search MCP tools exposed by the tool gateway."""

from __future__ import annotations

from typing import Any

from TOOLS._mcp_gateway import call_gateway_tool
from TOOLS._helpers import normalize_timeout

TOOL_NAME = "mcp_search_tools"
TOOL_DESCRIPTION = "Search MCP tools by query and optional group (all/read/write/destructive)."
SEARCH_GROUPS = ("all", "read", "write", "destructive")
DEFAULT_DISCOVERY_QUERY = "list available tools"
GROUP_DEFAULT_QUERY = {
    "all": DEFAULT_DISCOVERY_QUERY,
    "read": "read list get fetch show describe inspect",
    "write": "create write update modify set save post put send",
    "destructive": "delete remove drop destroy wipe revoke terminate kill",
}
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Free-text intent or keywords (tool name, operation, or parameter names).",
        },
        "group": {
            "type": "string",
            "enum": list(SEARCH_GROUPS),
            "description": "Risk/capability group filter: all, read, write, destructive.",
        },
        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        "server": {"type": "string"},
        "force_refresh": {"type": "boolean"},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 180},
    },
    "additionalProperties": False,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    group = _normalize_group(arguments.get("group"))
    query = _resolve_query(arguments.get("query"), group=group)

    payload: dict[str, Any] = {"query": query}
    if "limit" in arguments:
        payload["limit"] = int(arguments["limit"])
    if "server" in arguments and str(arguments.get("server", "")).strip():
        payload["server"] = str(arguments["server"]).strip()
    if "force_refresh" in arguments:
        payload["forceRefresh"] = bool(arguments["force_refresh"])

    timeout = normalize_timeout(arguments, default=60, maximum=180)
    try:
        raw = call_gateway_tool(
            repo_root=repo_root,
            gateway_tool="tool_search",
            payload=payload,
            timeout_seconds=timeout,
        )
    except Exception as error:
        return {"error": {"message": str(error) or "unknown error"}}

    if bool(raw.get("isError")):
        return {"error": _gateway_error(raw, default_message="tool_search failed")}

    structured = raw.get("structuredContent") if isinstance(raw, dict) else None
    source = structured if isinstance(structured, dict) else raw
    matches = source.get("matches") if isinstance(source, dict) else None
    if not isinstance(matches, list):
        matches = None
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

    if group != "all":
        normalized_tools = [tool for tool in normalized_tools if tool.get("risk") == group]

    return {
        "result": {
            "tools": normalized_tools,
            "count": len(normalized_tools),
            "query_used": query,
            "group_used": group,
            "raw": source if isinstance(source, dict) else {},
        }
    }


def _gateway_error(raw: dict[str, Any], *, default_message: str) -> dict[str, Any]:
    structured = raw.get("structuredContent")
    if isinstance(structured, dict):
        error_value = structured.get("error")
        if isinstance(error_value, dict):
            return error_value
        if isinstance(error_value, str) and error_value.strip():
            return {"message": error_value.strip()}
    content = raw.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    return {"message": text.strip()}
    return {"message": default_message}


def _normalize_group(raw_group: Any) -> str:
    if not isinstance(raw_group, str):
        return "all"
    normalized = raw_group.strip().lower()
    if normalized in SEARCH_GROUPS:
        return normalized
    return "all"


def _resolve_query(raw_query: Any, *, group: str) -> str:
    if isinstance(raw_query, str) and raw_query.strip():
        return raw_query.strip()
    return GROUP_DEFAULT_QUERY.get(group, DEFAULT_DISCOVERY_QUERY)
