# Repository Guidelines (Current Workspace)

## Source of Truth
- The active runtime codebase lives in `AIDZeroCode/`.
- Root-level files exist mainly as launchers/wrappers for convenience.
- If guidance conflicts, prefer the rules in this file plus module-specific `AGENTS.md` files.

## Project Layout
- `AIDZero.py`: root launcher for the runtime.
- `AIDZeroCode/`: canonical runtime implementation.
- `AIDZeroCode/CORE/`: engine, gateway, API, runtime wiring.
- `AIDZeroCode/UI/`: runnable UI modules (`UI/<name>/entrypoint.py`).
- `AIDZeroCode/LLMProviders/`: provider adapters.
- `AIDZeroCode/TOOLS/`: local tool plugins.
- `AIDZeroCode/MCP/tool-gateway/`: MCP gateway implementation.
- `AIDZeroCode/tests/`: automated tests.
- `Agents/`: runtime agent profiles and prompts used by the launcher.
- `SKILLS/`: Codex skills (`SKILLS/<skill>/SKILL.md`).

## Runtime Commands
```bash
uv run AIDZero.py
uv run AIDZero.py --request "Summarize this repository"
uv run AIDZero.py --agent default
uv run AIDZero.py --headless
```

## Headless Mode
- `--headless` always uses the `default` agent profile.
- Reads prompt from `Agents/default/HeadlessPrompt.txt`.
- Writes outputs to:
  - `Results/latest.txt`
  - `Results/result_YYYYMMDD_HHMMSS.txt`

## Terminology Guardrails
- `agent` means runtime profile under `Agents/<name>/` (`<name>.json`, `system_prompt.md`, `HeadlessPrompt.txt`).
- `skill` means Codex workflow docs/scripts under `SKILLS/<skill_name>/`.
- If the user asks to create or modify an `agent`, do not create a `skill` unless explicitly requested.
- For agent creation/edit tasks, load and follow `SKILLS/agent-creator/SKILL.md` before editing `Agents/`.

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
