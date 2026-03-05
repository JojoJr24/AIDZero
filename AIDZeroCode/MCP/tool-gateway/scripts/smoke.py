#!/usr/bin/env python3
"""Quick smoke test for the Python MCP gateway."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    launcher = repo_root / "MCP" / "run-tool-gateway.sh"
    params = StdioServerParameters(command=str(launcher), args=[])

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            search = await session.call_tool("tool_search", {"query": "list available tools", "limit": 3})
            search_error = await session.call_tool("tool_search", {"query": "", "limit": 0})
            describe_error = await session.call_tool("tool_describe", {"tool_id": "missing:tool"})

    print("tool_search:", search.model_dump(exclude_none=True))
    print("tool_search_error:", search_error.model_dump(exclude_none=True))
    print("tool_describe_error:", describe_error.model_dump(exclude_none=True))


if __name__ == "__main__":
    asyncio.run(main())
