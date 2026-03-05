# Tool Gateway Guidelines (v0.2.0)

## Scope
This folder contains the Python MCP gateway used by AIDZero MCP client tools.

## Entry Points
- `gateway_server.py`: MCP server exposing `tool_search`, `tool_describe`, `tool_call`.
- `scripts/gateway-call.py`: one-shot stdio client used by app-side tools.
- `scripts/smoke.py`: smoke validation helper.

## Design Constraints
- Keep stdio transport compatible with MCP clients.
- Never block forever on a single downstream MCP server.
- Preserve response shape expected by `TOOLS/_mcp_gateway.py`.
- Prefer additive changes; avoid breaking tool names or argument names.

## Compatibility Contract
Required gateway tools:
- `tool_search`
- `tool_describe`
- `tool_call`

Behavior guarantees:
- `tool_call` returns structured error payloads when downstream calls fail.
- `tool_search` remains lightweight and bounded by per-server timeouts.

## Safe Change Workflow
1. Update implementation.
2. Run focused tests: `pytest tests/test_mcp_tools.py tests/test_engine.py -q`.
3. Run smoke call:
   `.venv/bin/python MCP/tool-gateway/scripts/gateway-call.py --tool tool_search --payload '{"query":"list available tools","limit":3}'`
4. Verify no regressions in timeout handling.
