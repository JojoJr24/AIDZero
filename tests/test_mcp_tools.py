from __future__ import annotations

from TOOLS import mcp_call_tool, mcp_describe_tool, mcp_health, mcp_search_tools


def test_mcp_tools_forward_expected_gateway_names_and_payloads(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []

    def fake_call_gateway_tool(*, repo_root, gateway_tool, payload=None, timeout_seconds=60):
        calls.append(
            {
                "repo_root": repo_root,
                "gateway_tool": gateway_tool,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {"tool": gateway_tool, "payload": payload}

    monkeypatch.setattr(mcp_health, "call_gateway_tool", fake_call_gateway_tool)
    monkeypatch.setattr(mcp_search_tools, "call_gateway_tool", fake_call_gateway_tool)
    monkeypatch.setattr(mcp_describe_tool, "call_gateway_tool", fake_call_gateway_tool)
    monkeypatch.setattr(mcp_call_tool, "call_gateway_tool", fake_call_gateway_tool)

    mcp_health.run({}, repo_root=tmp_path, memory=None)
    mcp_search_tools.run({"query": "find browser", "limit": 3, "force_refresh": True}, repo_root=tmp_path, memory=None)
    mcp_describe_tool.run({"tool_id": "srv:tool", "force_refresh": True}, repo_root=tmp_path, memory=None)
    mcp_call_tool.run(
        {"tool_id": "srv:tool", "args": {"q": "x"}, "timeout_ms": 3000, "force_refresh": True},
        repo_root=tmp_path,
        memory=None,
    )

    assert [call["gateway_tool"] for call in calls] == [
        "tool_health",
        "tool_search",
        "tool_describe",
        "tool_call",
    ]
    assert calls[1]["payload"] == {"query": "find browser", "limit": 3, "forceRefresh": True}
    assert calls[2]["payload"] == {"tool_id": "srv:tool", "forceRefresh": True}
    assert calls[3]["payload"] == {
        "tool_id": "srv:tool",
        "args": {"q": "x"},
        "forceRefresh": True,
        "timeoutMs": 3000,
    }
