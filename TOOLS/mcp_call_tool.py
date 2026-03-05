"""Call one MCP tool through gateway by tool_id and args."""

from __future__ import annotations

import json
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
    try:
        raw = call_gateway_tool(
            repo_root=repo_root,
            gateway_tool="tool_call",
            payload=payload,
            timeout_seconds=timeout,
        )
    except Exception as error:
        return {"error": {"message": str(error) or "unknown error"}}

    if bool(raw.get("isError")):
        return {"error": _gateway_error(raw, default_message="tool_call failed")}

    structured = raw.get("structuredContent")
    forwarded_output = _extract_forwarded_output(raw, structured if isinstance(structured, dict) else None)
    metadata = _extract_call_metadata(structured if isinstance(structured, dict) else None)
    if forwarded_output is not None:
        payload: dict[str, Any] = {"result": forwarded_output}
        if metadata:
            payload["meta"] = metadata
        return payload
    if isinstance(structured, dict):
        payload = {"result": structured}
        if metadata:
            payload["meta"] = metadata
        return payload
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


def _extract_forwarded_output(raw: dict[str, Any], structured: dict[str, Any] | None) -> Any | None:
    if isinstance(structured, dict):
        forwarded = structured.get("forwardedResult")
        if isinstance(forwarded, dict) and forwarded.get("structuredContent") is not None:
            return forwarded.get("structuredContent")

    texts = _extract_text_blocks(raw.get("content"))
    if not texts:
        return None
    if _looks_like_gateway_summary(texts[0]):
        texts = texts[1:]
    if not texts:
        return None

    parsed_blocks = [_parse_json_if_possible(text) for text in texts]
    if len(parsed_blocks) == 1:
        return parsed_blocks[0]
    return parsed_blocks


def _extract_call_metadata(structured: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(structured, dict):
        return {}
    meta: dict[str, Any] = {}
    if "toolId" in structured:
        meta["tool_id"] = structured.get("toolId")
    if "server" in structured:
        meta["server"] = structured.get("server")
    if "tool" in structured:
        meta["tool"] = structured.get("tool")
    if "risk" in structured:
        meta["risk"] = structured.get("risk")
    if "args" in structured:
        meta["args"] = structured.get("args")
    forwarded = structured.get("forwardedResult")
    if isinstance(forwarded, dict) and "isError" in forwarded:
        meta["forwarded_is_error"] = bool(forwarded.get("isError"))
    return {key: value for key, value in meta.items() if value is not None}


def _extract_text_blocks(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            out.append(text.strip())
    return out


def _looks_like_gateway_summary(text: str) -> bool:
    normalized = text.strip()
    return normalized.startswith("Called ") and " -> " in normalized


def _parse_json_if_possible(text: str) -> Any:
    raw = text.strip()
    if not raw:
        return raw
    if raw[0] not in "{[":
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
