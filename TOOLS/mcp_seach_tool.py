"""Backward-compatible alias for MCP tool search.

Some clients call this tool with a historical typo (`mcp_seach_tool`).
Keep it available and delegate to `mcp_search_tools`.
"""

from __future__ import annotations

from typing import Any

from TOOLS import mcp_search_tools

TOOL_NAME = "mcp_seach_tool"
TOOL_DESCRIPTION = "Backward-compatible alias for mcp_search_tools."
TOOL_PARAMETERS = mcp_search_tools.TOOL_PARAMETERS


def run(arguments: dict[str, Any], *, repo_root, memory):
    return mcp_search_tools.run(arguments, repo_root=repo_root, memory=memory)
