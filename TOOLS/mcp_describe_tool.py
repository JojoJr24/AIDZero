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
    try:
        raw = call_gateway_tool(
            repo_root=repo_root,
            gateway_tool="tool_describe",
            payload=payload,
            timeout_seconds=timeout,
        )
    except Exception as error:
        return {"error": {"message": str(error) or "unknown error"}}

    if bool(raw.get("isError")):
        return {"error": _gateway_error(raw, default_message="tool_describe failed")}

    structured = raw.get("structuredContent")
    if isinstance(structured, dict):
        return {"result": structured}
    return {"result": raw}


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
