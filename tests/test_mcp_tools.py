from __future__ import annotations

from TOOLS import mcp_call_tool, mcp_describe_tool, mcp_health, mcp_seach_tool, mcp_search_tools


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
        if gateway_tool == "tool_search":
            return {
                "matches": [
                    {
                        "id": "srv:browser.open",
                        "server": "srv",
                        "tool": "browser.open",
                        "description": "Open browser",
                        "risk": "medium",
                    }
                ]
            }
        return {"tool": gateway_tool, "payload": payload}

    monkeypatch.setattr(mcp_health, "call_gateway_tool", fake_call_gateway_tool)
    monkeypatch.setattr(mcp_search_tools, "call_gateway_tool", fake_call_gateway_tool)
    monkeypatch.setattr(mcp_describe_tool, "call_gateway_tool", fake_call_gateway_tool)
    monkeypatch.setattr(mcp_call_tool, "call_gateway_tool", fake_call_gateway_tool)

    mcp_health.run({}, repo_root=tmp_path, memory=None)
    listed = mcp_search_tools.run({"limit": 3, "force_refresh": True}, repo_root=tmp_path, memory=None)
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
    assert calls[1]["payload"] == {"query": "list available tools", "limit": 3, "forceRefresh": True}
    assert listed["count"] == 1
    assert listed["tools"][0]["tool_id"] == "srv:browser.open"
    assert calls[2]["payload"] == {"tool_id": "srv:tool", "forceRefresh": True}
    assert calls[3]["payload"] == {
        "tool_id": "srv:tool",
        "args": {"q": "x"},
        "forceRefresh": True,
        "timeoutMs": 3000,
    }


def test_mcp_seach_tool_alias_delegates_to_mcp_search_tools(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_search_run(arguments, *, repo_root, memory):
        captured["arguments"] = arguments
        captured["repo_root"] = repo_root
        captured["memory"] = memory
        return {"tools": [], "count": 0}

    monkeypatch.setattr(mcp_search_tools, "run", fake_search_run)

    result = mcp_seach_tool.run({"limit": 2}, repo_root=tmp_path, memory=None)

    assert result == {"tools": [], "count": 0}
    assert captured["arguments"] == {"limit": 2}
    assert captured["repo_root"] == tmp_path
    assert captured["memory"] is None
