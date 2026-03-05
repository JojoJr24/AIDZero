from __future__ import annotations

from TOOLS import mcp_call_tool, mcp_describe_tool, mcp_search_tools


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

    monkeypatch.setattr(mcp_search_tools, "call_gateway_tool", fake_call_gateway_tool)
    monkeypatch.setattr(mcp_describe_tool, "call_gateway_tool", fake_call_gateway_tool)
    monkeypatch.setattr(mcp_call_tool, "call_gateway_tool", fake_call_gateway_tool)

    listed = mcp_search_tools.run({"limit": 3, "force_refresh": True}, repo_root=tmp_path, memory=None)
    described = mcp_describe_tool.run({"tool_id": "srv:tool", "force_refresh": True}, repo_root=tmp_path, memory=None)
    called = mcp_call_tool.run(
        {"tool_id": "srv:tool", "args": {"q": "x"}, "timeout_ms": 3000, "force_refresh": True},
        repo_root=tmp_path,
        memory=None,
    )

    assert [call["gateway_tool"] for call in calls] == [
        "tool_search",
        "tool_describe",
        "tool_call",
    ]
    assert calls[0]["payload"] == {"query": "list available tools", "limit": 3, "forceRefresh": True}
    assert listed["result"]["count"] == 1
    assert listed["result"]["tools"][0]["tool_id"] == "srv:browser.open"
    assert listed["result"]["group_used"] == "all"
    assert described["result"]["tool"] == "tool_describe"
    assert called["result"]["tool"] == "tool_call"
    assert calls[1]["payload"] == {"tool_id": "srv:tool", "forceRefresh": True}
    assert calls[2]["payload"] == {
        "tool_id": "srv:tool",
        "args": {"q": "x"},
        "forceRefresh": True,
        "timeoutMs": 3000,
    }


def test_mcp_search_tools_supports_query_and_group_filter(monkeypatch, tmp_path):
    captured_payloads: list[dict[str, object]] = []

    def fake_call_gateway_tool(*, repo_root, gateway_tool, payload=None, timeout_seconds=60):
        del repo_root, timeout_seconds
        captured_payloads.append(payload or {})
        assert gateway_tool == "tool_search"
        return {
            "matches": [
                {
                    "id": "srv:files.read",
                    "server": "srv",
                    "tool": "files.read",
                    "description": "Read files",
                    "risk": "read",
                },
                {
                    "id": "srv:files.delete",
                    "server": "srv",
                    "tool": "files.delete",
                    "description": "Delete files",
                    "risk": "destructive",
                },
            ]
        }

    monkeypatch.setattr(mcp_search_tools, "call_gateway_tool", fake_call_gateway_tool)

    result = mcp_search_tools.run(
        {"group": "read", "query": "filesystem", "limit": 10},
        repo_root=tmp_path,
        memory=None,
    )

    assert captured_payloads == [{"query": "filesystem", "limit": 10}]
    assert result["result"]["group_used"] == "read"
    assert result["result"]["count"] == 1
    assert result["result"]["tools"][0]["tool_id"] == "srv:files.read"


def test_mcp_call_tool_unwraps_forwarded_structured_content(monkeypatch, tmp_path):
    def fake_call_gateway_tool(*, repo_root, gateway_tool, payload=None, timeout_seconds=60):
        del repo_root, payload, timeout_seconds
        assert gateway_tool == "tool_call"
        return {
            "isError": False,
            "structuredContent": {
                "toolId": "deep-research:deep_research",
                "server": "deep-research",
                "tool": "deep_research",
                "risk": "read",
                "args": {"query": "ultimas noticias"},
                "forwardedResult": {
                    "contentBlocks": 1,
                    "structuredContent": {
                        "query": "ultimas noticias",
                        "results": [{"title": "n1", "url": "https://example.com/n1"}],
                    },
                    "isError": False,
                },
            },
            "content": [
                {
                    "type": "text",
                    "text": "Called deep-research:deep_research (deep-research/deep_research) with {} -> ok",
                }
            ],
        }

    monkeypatch.setattr(mcp_call_tool, "call_gateway_tool", fake_call_gateway_tool)

    result = mcp_call_tool.run(
        {"tool_id": "deep-research:deep_research", "args": {"query": "ultimas noticias"}},
        repo_root=tmp_path,
        memory=None,
    )

    assert result["result"]["query"] == "ultimas noticias"
    assert result["result"]["results"][0]["title"] == "n1"
    assert result["meta"]["tool_id"] == "deep-research:deep_research"
    assert result["meta"]["tool"] == "deep_research"


def test_mcp_call_tool_unwraps_forwarded_json_text_block(monkeypatch, tmp_path):
    def fake_call_gateway_tool(*, repo_root, gateway_tool, payload=None, timeout_seconds=60):
        del repo_root, payload, timeout_seconds
        assert gateway_tool == "tool_call"
        return {
            "isError": False,
            "structuredContent": {
                "toolId": "deep-research:deep_research",
                "server": "deep-research",
                "tool": "deep_research",
                "risk": "read",
                "args": {"query": "ultimas noticias"},
                "forwardedResult": {
                    "contentBlocks": 1,
                    "structuredContent": None,
                    "isError": False,
                },
            },
            "content": [
                {
                    "type": "text",
                    "text": "Called deep-research:deep_research (deep-research/deep_research) with {} -> ok",
                },
                {
                    "type": "text",
                    "text": "{\"query\":\"ultimas noticias\",\"results\":[{\"title\":\"n1\"}]}",
                },
            ],
        }

    monkeypatch.setattr(mcp_call_tool, "call_gateway_tool", fake_call_gateway_tool)

    result = mcp_call_tool.run(
        {"tool_id": "deep-research:deep_research", "args": {"query": "ultimas noticias"}},
        repo_root=tmp_path,
        memory=None,
    )

    assert result["result"]["query"] == "ultimas noticias"
    assert result["result"]["results"][0]["title"] == "n1"
    assert result["meta"]["server"] == "deep-research"
