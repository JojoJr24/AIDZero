# AIDZero

**An agent that generates agents for production environments by forking itself.**  
**AgentID = 0**

AIDZero is the root agent in a recursive agent-building system. It analyzes a user goal, forks a child workspace by copying only parent environment folders, and injects a new generated `agent/` runtime plus task-specific logic.

## Core Idea

- `AIDZero` (AgentID `0`) is the origin agent.
- New agents are generated as controlled forks from AIDZero itself.
- The fork copies only: `LLMProviders`, `MCP`, `SKILLS`, `TOOLS`, `UI`, `.aidzero`.
- The child receives a fresh runtime `agent/` package and regenerated `main.py`.
- The generated agent receives a workspace-safe `main.py` entrypoint with prompt-driven file actions constrained to its own folder.
- The generated `agent/child_manifest.json` stores the user request and mission context used for the new agent.
- The generated agent receives `agent_config.json` for runtime provider/model selection.
- The generated `main.py` is expected to support production execution patterns (including cron-style runs and output suitable for chat forwarding).

## Repository Layout

- `AIDZero.py`: root runtime entrypoint.
- `main.py`: compatibility wrapper to the root entrypoint.
- `agent/`: planning, cataloging, entrypoint generation, and scaffolding core.
- `LLMProviders/`: provider implementations (`AID-*` folders only tracked).
- `SKILLS/`: skill modules (`AID-*` folders only tracked).
- `MCP/`: MCP integrations (`AID-*` folders only tracked).
- `TOOLS/`: tool modules (`AID-*` folders only tracked).
- `UI/`: user interfaces (current terminal interface included).

## Quick Start

```bash
uv sync
uv run AIDZero.py --help
```

First run is interactive:
1. Select UI
2. Select provider
3. Select model

Configuration is saved to:

```text
.aidzero/runtime_config.json
```

You can force setup again with:

```bash
uv run AIDZero.py --reconfigure
```

Launch the Web UI directly:

```bash
uv run AIDZero.py --ui web --ui-option host=127.0.0.1 --ui-option port=8787
```

## Typical Usage

List available runtime options:

```bash
uv run AIDZero.py --list-options
```

Create a plan only (no scaffold):

```bash
uv run AIDZero.py --request "Build an agent for daily KPI reports" --dry-run
```

Generate the new agent project:

```bash
uv run AIDZero.py --request "Build an agent for daily KPI reports" --yes
```

## MCP Servers

The MCP gateway reads server definitions from:

```text
.aidzero/mcporter.json
```

If a legacy file exists at `MCP/AID-tool-gateway/config/mcporter.json`, the gateway migrates it automatically on startup.

Add or edit servers under the `mcpServers` object. Example:

```json
{
  "mcpServers": {
    "filesystem": {
      "description": "Local filesystem MCP server",
      "command": [
        "uv",
        "run",
        "python",
        "server.py"
      ]
    },
    "linear": {
      "description": "Hosted MCP server",
      "url": "https://mcp.linear.app/mcp"
    }
  }
}
```

Then run:

```bash
bash run-tool-gateway.sh
cd MCP/AID-tool-gateway && node scripts/smoke.mjs
```

For command-based servers, paths are resolved relative to `.aidzero/` because the config file lives there.

## Design Principles

- Modular by default.
- Production-oriented output.
- LLM-driven planning and entrypoint generation.
- Minimal dependency footprint for generated bootstrap code.
- Keep generated changes constrained to the cloned workspace only.
