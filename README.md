# AIDZero Runtime (v0.2.0)

AIDZero is a modular runtime for LLM agents with dynamic UI loading, pluggable provider adapters, local/MCP tools, and agent profiles.

## What This App Does

- Runs an agent loop with tool calling and optional streaming.
- Loads runnable UIs dynamically from `UI/<name>/entrypoint.py`.
- Supports multiple model providers through `LLMProviders/` adapters.
- Uses `Agents/*.json` profiles to control runtime defaults and feature flags.
- Persists runtime artifacts under `.aidzero/` (history, outputs, memory, active profile).
- Integrates an MCP tool gateway (`tool_search`, `tool_describe`, `tool_call`).

## Key Runtime Concepts

- Trigger source: interactive UI events collected through `core/gateway.py`.
- Tool priority policy:
  1. Local tools (`TOOLS/*.py`)
  2. Skills (`SKILLS/` via skill tools)
  3. MCP gateway tools
- History and memory can be enabled/disabled per profile.

## Run

```bash
# Main launcher (uses active profile defaults)
uv run AIDZero.py
uv run AIDZero.py --request "Summarize this repository"
uv run AIDZero.py --agent default --request "List available tools"

# Split runtime: core only
uv run aidzero-core --agent default --host 0.0.0.0 --port 8765

# Split runtime: UI only (connect to existing core)
uv run aidzero-ui --ui tui --core-url http://127.0.0.1:8765

# All in one (core + UI)
uv run aidzero-all --ui tui --agent default --host 127.0.0.1 --port 8765
```

## Project Layout

- `AIDZero.py`: root runtime launcher.
- `core/`: engine, gateway, API server/client, split launchers, runtime wiring.
- `UI/tui/`: Textual UI entrypoint and app runtime.
- `LLMProviders/`: provider adapters (OpenAI, Claude, Gemini, local providers, etc.).
- `TOOLS/`: local tool plugins (`TOOL_NAME`, schema, `run(...)`).
- `MCP/tool-gateway/`: Python MCP gateway implementation.
- `Agents/`: profile and prompt definitions.
- `tests/`: test suite.

## MCP Gateway Quick Check

```bash
bash MCP/run-tool-gateway.sh
.venv/bin/python MCP/tool-gateway/scripts/gateway-call.py --tool tool_search --payload '{"query":"list available tools","limit":3}'
```

## Profiles

Profiles are loaded from `Agents/*.json` and persist active selection in `.aidzero/agent_profile.json`.

Each profile can define:
- `system_prompt_file` or inline `system_prompt`
- runtime defaults: `ui`, `provider`, `model`
- features: `memory`, `history`
- module allowlists: `tools`, `dash`

## Outputs and State

- `.aidzero/store/history.jsonl`
- `.aidzero/store/output.jsonl`
- `.aidzero/output/latest.txt`
- `.aidzero/memory.json`
- `.aidzero/agent_profile.json`
