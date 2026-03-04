# AIDZero Runtime

This repository contains a modular LLM agent runtime built for AIDZero.

## Implemented Architecture

- **Dynamic UI runtime**
  - UIs are loaded dynamically from `UI/<name>/entrypoint.py`
  - each UI entrypoint exposes `run_ui(...)`
- **Gateway triggers**
  - `interactive` prompts from terminal/TUI
- **Injected every turn**
  - system prompt
  - tool schemas
  - JSONL history
  - memory snapshot
- **Tool runtime**
  - tools are loaded dynamically from `TOOLS/*.py` (one `.py` file per tool)
  - sandbox command execution
  - file read/write/list tools
  - persistent memory tools
  - computer-control tool (open_url, type_text, key_press, mouse, screenshot, run)
  - MCP gateway client tools (`mcp_health`, `mcp_search_tools`, `mcp_describe_tool`, `mcp_call_tool`)
  - skill tools (`list_skills`, `read_skill`, `run_skill_script`)
- **Outputs**
  - `.aidzero/store/history.jsonl`
  - `.aidzero/store/output.jsonl`
  - `.aidzero/output/latest.txt`
  - `.aidzero/memory.json`

## Run

```bash
uv run AIDZero.py --list-options
uv run AIDZero.py --provider openai --model gpt-4o-mini --ui terminal
uv run AIDZero.py --provider openai --model gpt-4o-mini --ui terminal --request "Summarize repo"
uv run AIDZero.py --request "Summarize repo"
```

## Split UI/Core (different port or IP)

Three launchers are now available:

```bash
# 1) Core only
uv run aidzero-core --agent default --host 0.0.0.0 --port 8765

# 2) UI only (connects to an existing core)
uv run aidzero-ui --ui terminal --core-url http://127.0.0.1:8765
uv run aidzero-ui --ui tui --core-url http://192.168.1.20:8765

# 3) All-in-one launcher (starts core + UI as separate processes)
uv run aidzero-all --ui terminal --agent default --host 127.0.0.1 --port 8765
```

`aidzero-all` starts `core` first, waits for `/health`, then launches the selected UI connected through `core_url`.

## Runtime Setup (TUI)

From the TUI you can inspect/update runtime values in the active profile:

```text
/setup show
/setup runtime terminal openai gpt-4o-mini
```

## Agent Profiles (`Agents/*.json`)

- Runtime profiles are loaded from `Agents/*.json`.
- Active profile is persisted in `.aidzero/agent_profile.json`.
- Each profile can define:
  - `system_prompt_file` or `system_prompt`
  - `system_prompt_file` must point to a file inside `Agents/`
  - `runtime.ui`, `runtime.provider`, `runtime.model`
  - `features.memory`: `true|false` to enable/disable memory tools/store
  - `features.history`: `true|false` to enable/disable runtime + prompt history
  - `modules.tools`: `"all"` or a list of tool names (`TOOL_NAME`)
  - `modules.dash`: `"all"` or a list of DASH module names (`DASH/<name>.py`)

Included examples:

- `Agents/default.json`: current default runtime behavior
- `Agents/planificador.json`: planning-focused profile with restricted tools

Switch profile inside the TUI with:

```text
/agent
/agent list
/agent planificador
/agent default
```

## Tool Call Format

The LLM must call tools with:

```text
<AID_TOOL_CALL>{"name":"sandbox_run","arguments":{"command":"ls -la"}}</AID_TOOL_CALL>
```

## Tool Plugin Contract

Each file in `TOOLS/*.py` must expose:

- `TOOL_NAME: str`
- `TOOL_DESCRIPTION: str`
- `TOOL_PARAMETERS: dict` (JSON-schema-like params)
- `run(arguments: dict, *, repo_root: Path, memory: MemoryStore) -> Any`
