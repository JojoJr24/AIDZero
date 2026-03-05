# Repository Guidelines (Current Workspace)

## Source of Truth
- The active runtime codebase lives in `AIDZeroCode/`.
- Root-level files exist mainly as launchers/wrappers for convenience.
- If guidance conflicts, prefer the rules in this file plus module-specific `AGENTS.md` files.

## Project Layout
- `AIDZero.py`: root launcher for the runtime.
- `AIDZeroCode/`: canonical runtime implementation.
- `AIDZeroCode/core/`: engine, gateway, API, runtime wiring.
- `AIDZeroCode/UI/`: runnable UI modules (`UI/<name>/entrypoint.py`).
- `AIDZeroCode/LLMProviders/`: provider adapters.
- `AIDZeroCode/TOOLS/`: local tool plugins.
- `AIDZeroCode/MCP/tool-gateway/`: MCP gateway implementation.
- `AIDZeroCode/tests/`: automated tests.
- `Agents/`: runtime profiles and system prompts used by the launcher.

## Runtime Commands
```bash
uv run AIDZero.py
uv run AIDZero.py --request "Summarize this repository"
uv run AIDZero.py --agent default
uv run AIDZero.py --headless
```

## Headless Mode
- `--headless` always uses the `default` agent profile.
- Reads prompt from `HeadlessPrompt.txt` in repo root.
- Writes outputs to:
  - `Results/latest.txt`
  - `Results/result_YYYYMMDD_HHMMSS.txt`

## Development Rules
- Keep code/comments/docs in English.
- Use `uv` for Python commands.
- Add or update tests for behavior changes.
- Prefer focused, additive changes over broad rewrites.
- Do not break CLI contracts without updating docs and tests together.

## Validation Baseline
Run relevant subsets when making changes:
```bash
uv run --with pytest pytest AIDZeroCode/tests -q
```
