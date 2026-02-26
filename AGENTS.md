# Repository Guidelines

## Project Structure & Module Organization

- Root contains Python bootstrap files (`pyproject.toml`, `hello.py`) and repository-level docs.
- `LLMProviders/` stores provider adapters (OpenAI-compatible, local backends like Ollama/LM Studio/llama.cpp). Keep one adapter per subfolder.
- `TOOLS/` stores reusable tool wrappers and tool contracts.
- `SKILLS/` stores reusable agent skills. Each skill folder should include `SKILL.md`, optional `references/`, and optional `scripts/`.
- `MCP/AID-tool-gateway/` contains the JavaScript MCP gateway.
- `MCP/AID-tool-gateway/src/` holds runtime code and `MCP/AID-tool-gateway/scripts/` stores operational checks.
- `MCP/mcporter.json` stores MCP server configuration consumed by the gateway runtime.
- `agent/` contains the core orchestration agent that scans components, asks the LLM for requirements, and scaffolds new agent projects.
- `UI/AGENTS.md` defines baseline UX/runtime requirements for every UI implementation.
- Add new Python core modules under `src/aidzero/` and tests under `tests/`.

## Build, Test, and Development Commands

```bash
uv sync                                               # Install/update Python dependencies
uv run hello.py                                       # Run Python bootstrap entrypoint
uv run pytest                                         # Run Python tests
uv run AIDZero.py --help                              # Root entrypoint (UI/provider/model selection)
cd MCP/AID-tool-gateway && npm install                # Install gateway dependencies
cd MCP/AID-tool-gateway && npm start                  # Start MCP gateway (stdio)
cd MCP/AID-tool-gateway && npm run dev                # Start gateway in dev mode
cd MCP/AID-tool-gateway && node scripts/smoke.mjs     # Run gateway smoke test
```

Use `uv` for all Python dependency/runtime commands and keep script paths repo-relative.

## Coding Style & Naming Conventions

- Use English for all code, comments, docs, commit messages, and PR text.
- Python: PEP 8, 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and type hints for public APIs.
- JavaScript (gateway): ESM modules, `camelCase` identifiers, and `UPPER_SNAKE_CASE` constants.
- Keep modules composable and isolated: provider adapters, tool adapters, skill registry, orchestration core, and MCP integration must not be tightly coupled.

## Testing Guidelines

- Current executable check is `MCP/AID-tool-gateway/scripts/smoke.mjs`.
- For new features, add automated tests.
- Python tests should use `tests/test_<module>.py` (pytest style).
- JavaScript tests should use `<module>.test.mjs` or integration checks in `scripts/`.
- Cover success paths, schema validation errors, provider fallback logic, and tool/MCP failure handling.
- Target at least 80% coverage for core orchestration and routing modules.

## Commit & Pull Request Guidelines

- There is no established commit history yet; adopt Conventional Commits now (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Keep commits focused and atomic.
- PRs should include scope, rationale, test evidence (commands + key output), related issue/task, and config/environment impact.
- Include logs when changing operator-facing behavior (CLI output, tool routing, gateway responses, or provider selection flow).

## UI Naming Normalization (Mandatory)

- This rule applies to every UI surface (terminal, web, future UI modules) when displaying providers, tools, skills, MCP, UI names, or similar identifiers.
- If a displayed value starts with `AID-`, remove that prefix in the UI label.
- If a displayed value does not start with `AID-`, append ` (test)` to the UI label.
- Exception: model names must be displayed exactly as-is (no prefix removal and no ` (test)` suffix).
- Keep internal IDs unchanged for runtime logic, API calls, storage, and CLI arguments. Normalize only display labels.
- Reuse `agent.ui_display.to_ui_label()` and `agent.ui_display.to_ui_model_label()` instead of duplicating custom formatting logic.
- Examples:
  - `AID-claude` -> `claude`
  - `AID-google_gemini` -> `google_gemini`
  - `terminal` -> `terminal (test)`
  - `gpt-4o-mini` -> `gpt-4o-mini`

## UI Runtime Isolation (Mandatory)

- `AIDZero.py` must not import specific UI implementations directly.
- Runnable UIs are discovered dynamically from `UI/<ui_name>/entrypoint.py`.
- Every runnable UI must expose a function `run_ui(...)` in its `entrypoint.py`.
- UI-specific runtime settings must be handled inside each UI module, using shared CLI transport via repeated `--ui-option KEY=VALUE`.
- Keep UI module internals isolated: each UI decides how to parse and apply `ui_options` and must not require root-level UI-specific flags.

## UI Prompt History (Mandatory)

- Terminal and Web UIs must persist prompt history between runs.
- Terminal and Web UIs must provide a built-in way to select prompts from that history.
