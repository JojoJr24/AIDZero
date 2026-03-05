from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys


def _load_tool_index_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "MCP" / "tool-gateway" / "tool_index.py"
    spec = importlib.util.spec_from_file_location("aidzero_tool_index", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeRuntime:
    def list_servers(self):
        return ["filesystem"]

    async def list_tools(self, server_name, include_schema=True, timeout_seconds=8.0):
        del include_schema, timeout_seconds
        assert server_name == "filesystem"
        return [
            {
                "name": "read_file",
                "description": "Read file contents",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "delete_file",
                "description": "Delete file by path",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        ]


def test_tool_index_list_query_returns_indexed_tools():
    module = _load_tool_index_module()
    index = module.ToolIndex(_FakeRuntime())

    async def _run():
        result = await index.search("list available tools", limit=10)
        ids = [item["id"] for item in result["matches"]]
        assert ids == ["filesystem:delete_file", "filesystem:read_file"]

    asyncio.run(_run())


def test_tool_index_wildcard_query_returns_indexed_tools():
    module = _load_tool_index_module()
    index = module.ToolIndex(_FakeRuntime())

    async def _run():
        result = await index.search("*", limit=10)
        ids = [item["id"] for item in result["matches"]]
        assert ids == ["filesystem:delete_file", "filesystem:read_file"]

    asyncio.run(_run())
