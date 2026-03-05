# MCP Folder Guidelines (v0.2.0)

## Scope
This folder contains MCP runtime configuration and the gateway launcher used by AIDZero tools.

## Files and Responsibilities
- `mcporter.json`: source of truth for configured MCP servers.
- `run-tool-gateway.sh`: stdio launcher used by app/tool callers.
- `tool-gateway/`: Python MCP gateway implementation.

## Editing Rules
- Keep `mcporter.json` valid JSON and preserve top-level shape: `{ "mcpServers": { ... } }`.
- Prefer absolute paths for server working directories.
- For command-based servers, ensure commands work non-interactively.
- Keep `run-tool-gateway.sh` stable: it must `exec` and keep stdio open.
- Do not introduce blocking prompts in launcher scripts.

## Operational Expectations
- Gateway startup must be deterministic from repository root.
- Failing MCP servers must not block the whole gateway indefinitely.
- Timeouts should fail fast with explicit error messages.

## Validation Checklist
After changes in this folder, verify:
1. `bash MCP/run-tool-gateway.sh` starts without immediate crash.
2. `tool_search` call succeeds through `MCP/tool-gateway/scripts/gateway-call.py`.
3. `tool_search` returns quickly even if one MCP server is down.
