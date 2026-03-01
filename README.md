# AIDZero OpenClaw-Style Runtime

This repository now contains a fresh implementation of an OpenClaw-like LLM agent runtime.

## Implemented Architecture

- **Dynamic UI runtime**
  - UIs are loaded dynamically from `UI/*.py`
  - each UI module exposes `run_ui(...)`
- **Gateway triggers**
  - `heartbeat` from `HEARTBEAT.md`
  - `cron` from `.aidzero/cron_prompt.txt`
  - `messengers` inbox from `.aidzero/inbox/messages.jsonl`
  - `webhooks` inbox from `.aidzero/inbox/webhooks.jsonl`
  - `interactive` prompts from terminal UI
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
uv run AIDZero.py --provider openai --model gpt-4o-mini --ui terminal --trigger all
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
