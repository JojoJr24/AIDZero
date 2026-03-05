#!/usr/bin/env python3
"""Call one gateway tool over stdio and print JSON output."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call MCP tool gateway")
    parser.add_argument("--tool", required=True)
    parser.add_argument("--payload", default="{}")
    return parser.parse_args()


async def run(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    gateway_launcher = repo_root / "MCP" / "run-tool-gateway.sh"

    params = StdioServerParameters(command=str(gateway_launcher), args=[])
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, payload)

    content = []
    for block in result.content:
        content.append(block.model_dump(exclude_none=True))

    output: dict[str, Any] = {
        "tool": tool_name,
        "isError": bool(result.isError),
        "structuredContent": result.structuredContent,
        "content": content,
    }

    if output["structuredContent"] is None:
        parsed = _extract_json_from_text_content(content)
        if isinstance(parsed, dict):
            output["structuredContent"] = parsed
    return output


def _extract_json_from_text_content(content: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(content) != 1:
        return None
    block = content[0]
    if block.get("type") != "text" or not isinstance(block.get("text"), str):
        return None
    raw = block["text"].strip()
    if not raw.startswith("{"):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def main() -> None:
    args = parse_args()
    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON passed to --payload: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit("--payload must be a JSON object")

    result = asyncio.run(run(args.tool.strip(), payload))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
