#!/usr/bin/env python3
"""MCP tool gateway implemented in Python."""

from __future__ import annotations

import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from runtime import McpRuntime
from schema_validator import SchemaValidator
from tool_index import ToolIndex


GATEWAY_NAME = "tool-search-gateway"
GATEWAY_VERSION = "0.1.0"
MCP_CONFIG_FILENAME = "mcporter.json"
DEFAULT_MCPORTER_CONFIG = {"mcpServers": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AIDZero MCP tool gateway")
    parser.add_argument("transport", nargs="?", default="stdio", choices=["stdio"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = ensure_mcporter_config_path(repo_root)

    runtime = McpRuntime(
        config_path=config_path,
        client_name=f"{GATEWAY_NAME}:runtime",
        client_version=GATEWAY_VERSION,
    )
    tool_index = ToolIndex(
        runtime,
        exclude_servers=[GATEWAY_NAME],
        server_list_timeout_seconds=read_env_float("AID_MCP_LIST_TIMEOUT_SECONDS", 8.0),
    )
    schema_validator = SchemaValidator()

    mcp = FastMCP(
        name=GATEWAY_NAME,
        instructions=(
            "tool_search narrows MCP tools by intent, tool_describe reveals one schema, "
            "tool_call executes through configured MCP servers."
        ),
    )

    @mcp.tool(name="tool_search", description="Return up to 5 MCP tools ranked by textual relevance.")
    async def tool_search(
        query: str,
        limit: int = 5,
        server: str | None = None,
        forceRefresh: bool = False,
    ) -> CallToolResult:
        try:
            result = await tool_index.search(
                query,
                limit=limit,
                server_hint=server,
                force_refresh=forceRefresh,
            )
            text = format_search_text(query, result.get("matches", []))
            return success_result(text=text, structured=result)
        except Exception as error:
            message = str(error) or "unknown error"
            return error_result(text=f'tool_search failed for "{query}": {message}', structured={"error": message})

    @mcp.tool(name="tool_describe", description="Return stored schema and metadata for a tool_id.")
    async def tool_describe(tool_id: str, forceRefresh: bool = False) -> CallToolResult:
        try:
            result = await tool_index.describe(tool_id, force_refresh=forceRefresh)
            text = format_describe_text(result)
            return success_result(text=text, structured=result)
        except Exception as error:
            message = str(error) or "unknown error"
            return error_result(
                text=f"tool_describe failed for {tool_id or 'unknown'}: {message}",
                structured={"toolId": tool_id.strip() if isinstance(tool_id, str) else "", "error": message},
            )

    @mcp.tool(name="tool_call", description="Execute a tool previously returned by tool_search/tool_describe.")
    async def tool_call(
        tool_id: str,
        args: dict[str, Any] | None = None,
        timeoutMs: int | None = None,
        forceRefresh: bool = False,
    ) -> CallToolResult:
        forwarded_args = ensure_args_object(args)
        timeout_value = normalize_timeout(timeoutMs)
        entry = None
        try:
            entry = await tool_index.get_tool(tool_id, force_refresh=forceRefresh)
            prepared_args = schema_validator.validate(entry.id, entry.input_schema, forwarded_args)
            forwarded = await runtime.call_tool(
                entry.server,
                entry.tool,
                args=prepared_args,
                timeout_ms=timeout_value,
            )

            content = _to_content_blocks(forwarded.get("content"))
            summary = build_call_summary(
                tool_id=entry.id,
                server_name=entry.server,
                tool_name=entry.tool,
                forwarded_args=prepared_args,
                is_error=bool(forwarded.get("isError", False)),
            )
            structured = {
                "toolId": entry.id,
                "server": entry.server,
                "tool": entry.tool,
                "risk": entry.risk,
                "args": prepared_args,
                "forwardedResult": {
                    "contentBlocks": len(content),
                    "structuredContent": forwarded.get("structuredContent"),
                    "isError": bool(forwarded.get("isError", False)),
                },
            }
            return CallToolResult(
                content=[TextContent(type="text", text=summary), *content],
                structuredContent=structured,
                isError=bool(forwarded.get("isError", False)),
            )
        except Exception as error:
            message = str(error) or "unknown error"
            fallback_tool_id = tool_id.strip() if isinstance(tool_id, str) else ""
            structured = {
                "toolId": fallback_tool_id,
                "server": getattr(entry, "server", None),
                "tool": getattr(entry, "tool", None),
                "risk": getattr(entry, "risk", None),
                "args": forwarded_args,
                "error": message,
            }
            return CallToolResult(
                content=[TextContent(type="text", text=f"tool_call failed for {fallback_tool_id or 'unknown'}: {message}")],
                structuredContent=structured,
                isError=True,
            )

    with suppress(BrokenPipeError, KeyboardInterrupt):
        mcp.run(args.transport)


def success_result(*, text: str, structured: dict[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=structured,
        isError=False,
    )


def error_result(*, text: str, structured: dict[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=structured,
        isError=True,
    )


def _to_content_blocks(raw_content: Any) -> list[TextContent]:
    if not isinstance(raw_content, list):
        return []
    out: list[TextContent] = []
    for item in raw_content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        out.append(TextContent(type="text", text=text))
    return out


def format_search_text(query: str, matches: list[dict[str, Any]]) -> str:
    if not matches:
        return f'No tools matched "{query}".'
    header = f'Top {len(matches)} matches for "{query}":'
    rows: list[str] = []
    for index, match in enumerate(matches, start=1):
        description = str(match.get("description") or "sin descripción").strip()
        params = format_parameter_summary(match.get("parameters"))
        rows.append(
            f"{index}. {match.get('id')} [risk:{match.get('risk')}, score:{match.get('score')}] {description}{params}"
        )
    return "\n".join([header, *rows])


def format_parameter_summary(parameters: Any) -> str:
    if not isinstance(parameters, list) or not parameters:
        return ""
    parts: list[str] = []
    for item in parameters:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        required = "*" if item.get("required") else ""
        parts.append(f"{name}{required}")
    if not parts:
        return ""
    return " | params: " + ", ".join(parts)


def format_describe_text(description: dict[str, Any]) -> str:
    input_summary = description.get("inputSummary") if isinstance(description.get("inputSummary"), dict) else {}
    required = input_summary.get("required") if isinstance(input_summary.get("required"), list) else []
    required_text = ", ".join(str(item) for item in required if isinstance(item, str)) or "none"

    fields = input_summary.get("fields") if isinstance(input_summary.get("fields"), list) else []
    field_text = ", ".join(
        f"{field.get('name')}:{field.get('type')}"
        for field in fields
        if isinstance(field, dict) and isinstance(field.get("name"), str)
    )
    if not field_text:
        field_text = "no parameters"

    return f"{description.get('id')} [risk:{description.get('risk')}] - required: {required_text} | fields: {field_text}"


def build_call_summary(*, tool_id: str, server_name: str, tool_name: str, forwarded_args: dict[str, Any], is_error: bool) -> str:
    serialized = json.dumps(forwarded_args, ensure_ascii=False) if forwarded_args else "{}"
    status = "error" if is_error else "ok"
    return f"Called {tool_id} ({server_name}/{tool_name}) with {serialized} -> {status}"


def ensure_args_object(args: dict[str, Any] | None) -> dict[str, Any]:
    if args is None:
        return {}
    if not isinstance(args, dict):
        raise ValueError("tool_call args must be an object")
    return args


def normalize_timeout(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("timeoutMs must be a positive number when provided") from error
    if numeric <= 0:
        raise ValueError("timeoutMs must be a positive number when provided")
    return numeric


def ensure_mcporter_config_path(repo_root: Path) -> Path:
    mcp_dir = repo_root / "MCP"
    mcp_dir.mkdir(parents=True, exist_ok=True)
    config_path = mcp_dir / MCP_CONFIG_FILENAME
    if config_path.is_file():
        return config_path

    legacy_paths = [
        repo_root / ".aidzero" / MCP_CONFIG_FILENAME,
        repo_root / "MCP" / "tool-gateway" / "config" / MCP_CONFIG_FILENAME,
    ]
    for legacy in legacy_paths:
        if legacy.is_file():
            config_path.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
            return config_path

    config_path.write_text(f"{json.dumps(DEFAULT_MCPORTER_CONFIG, indent=2)}\n", encoding="utf-8")
    return config_path


def read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value <= 0:
        return default
    return value


if __name__ == "__main__":
    main()
