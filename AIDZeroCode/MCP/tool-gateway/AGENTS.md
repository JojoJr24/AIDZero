# Tool Gateway Guidelines (v0.3.0)

## Scope
This folder contains the Python MCP gateway used by runtime MCP client tools.

## Entry Points
- `gateway_server.py`: MCP server exposing `tool_search`, `tool_describe`, `tool_call`.
- `scripts/gateway-call.py`: one-shot stdio client.
- `scripts/smoke.py`: smoke validation helper.

## Compatibility Contract
Required gateway tools and stable names:
- `tool_search`
- `tool_describe`
- `tool_call`

Behavior guarantees:
- Return structured error payloads when downstream calls fail.
- Keep calls bounded with timeouts.
- Preserve response shape expected by MCP client tooling.

## Validation
```bash
uv run --with pytest pytest tests/test_mcp_tools.py tests/test_engine.py -q
```
