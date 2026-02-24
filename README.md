# AIDZero

**An agent that generates agents for production environments by forking itself.**  
**AgentID = 0**

AIDZero is the root agent in a recursive agent-building system. It analyzes a user goal, decides which capabilities are required, and creates a new agent project by reusing only the necessary modules (providers, skills, tools, MCP integrations, and UI).

## Core Idea

- `AIDZero` (AgentID `0`) is the origin agent.
- New agents are generated as derived projects from AIDZero's own modular components.
- The generated agent receives its own `main.py` entrypoint, produced by the LLM from prompt context.
- The generated `main.py` is expected to support production execution patterns (including cron-style runs and output suitable for chat forwarding).

## Repository Layout

- `AIDZero.py`: root runtime entrypoint.
- `main.py`: compatibility wrapper to the root entrypoint.
- `agent_creator/`: planning, cataloging, entrypoint generation, and scaffolding core.
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

## Design Principles

- Modular by default.
- Production-oriented output.
- LLM-driven planning and entrypoint generation.
- Minimal dependency footprint for generated bootstrap code.
- Reuse only what each generated agent needs.
