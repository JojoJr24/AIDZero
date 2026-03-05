# MCP Folder Guidelines (v0.3.0)

## Scope
This folder contains MCP runtime configuration and the gateway launcher used by runtime MCP tools.

## Files and Responsibilities
- `mcporter.json`: source of truth for configured MCP servers.
- `run-tool-gateway.sh`: stdio launcher used by runtime tool callers.
- `tool-gateway/`: Python MCP gateway implementation.

## Editing Rules
- Keep `mcporter.json` valid JSON with top-level shape `{ "mcpServers": { ... } }`.
- Prefer deterministic command/server setup.
- Keep launcher scripts non-interactive and stdio-safe.
- Ensure one failing MCP backend does not block the full gateway.

## Validation
```bash
uv run --with pytest pytest tests/test_mcp_tools.py tests/test_tool_gateway_index.py -q
```
