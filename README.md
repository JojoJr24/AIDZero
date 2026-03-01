# AIDZero OpenClaw-Style Runtime

This repository now contains a fresh implementation of an OpenClaw-like LLM agent runtime.

## Implemented Architecture

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
  - sandbox command execution
  - file read/write/list tools
  - persistent memory tools
  - computer-control placeholder tool
- **Outputs**
  - `.aidzero/store/history.jsonl`
  - `.aidzero/store/output.jsonl`
  - `.aidzero/output/latest.txt`
  - `.aidzero/memory.json`

## Run

```bash
uv run AIDZero.py --list-options
uv run AIDZero.py --provider AID-openai --model gpt-4o-mini --ui terminal
uv run AIDZero.py --provider AID-openai --model gpt-4o-mini --ui terminal --request "Summarize repo"
uv run AIDZero.py --provider AID-openai --model gpt-4o-mini --ui terminal --trigger all
```

## Tool Call Format

The LLM must call tools with:

```text
<AID_TOOL_CALL>{"name":"sandbox_run","arguments":{"command":"ls -la"}}</AID_TOOL_CALL>
```
