---
name: agent-creator
description: Create or modify AIDZero runtime agent profiles under Agents/. Use when the user asks to add or edit an agent (prompt, runtime, tools, memory, history, headless). Do not create a skill unless explicitly requested.
---

# Agent Creator

## Purpose

Implement agent profiles for this repository using the current folder contract under `Agents/`.

## Repository Contract

- Runtime agent definitions live in `Agents/`, not in `AIDZeroCode/`.
- Agent work and skill work are different:
  - `Agents/<agent_name>/...` for runtime agents
  - `SKILLS/<skill_name>/...` for Codex skills
- Each agent must use one dedicated folder:
  - `Agents/<agent_name>/<agent_name>.json`
  - `Agents/<agent_name>/system_prompt.md`
  - `Agents/<agent_name>/HeadlessPrompt.txt`
- The folder name, JSON filename, and JSON `"name"` should match.

## When Using This Skill

Follow this skill when the task involves any of these:

- creating a new agent
- renaming or reorganizing an agent
- changing an agent system prompt
- changing runtime defaults (`ui`, `provider`, `model`)
- changing allowed tools or dash modules
- changing `memory` or `history`
- changing headless prompt behavior

Do not use this skill when the user explicitly asked to create/modify a `skill`.

## Execution Steps

1. Inspect the current `Agents/` layout and existing profiles before editing.
2. Confirm the user wants an agent profile (not a skill). If ambiguous, ask.
3. If any required detail is ambiguous, ask before editing.
4. Collect missing runtime decisions in this strict order: `provider` -> `model` -> `ui`.
5. Create or update the target folder as `Agents/<agent_name>/`.
6. Add or update `<agent_name>.json`.
7. Add or update `system_prompt.md`.
8. Add or update `HeadlessPrompt.txt`.
9. Keep the JSON minimal and valid. Required shape:

```json
{
  "name": "agent_name",
  "description": "Short description",
  "system_prompt_file": "system_prompt.md",
  "features": {
    "memory": true,
    "history": true
  },
  "runtime": {
    "ui": "tui",
    "provider": "openai",
    "model": "gpt-4o-mini"
  },
  "modules": {
    "tools": "all",
    "dash": "all"
  }
}
```

## Field Rules

- `name`: non-empty string; keep it aligned with folder/file names.
- `description`: optional but recommended.
- `system_prompt_file`: prefer `"system_prompt.md"` over inline `system_prompt`.
- `features.memory` and `features.history`: booleans only.
- `runtime.ui`, `runtime.provider`, `runtime.model`: required non-empty strings.
- `modules.tools` and `modules.dash`: either `"all"` or an array of non-empty strings.

## Clarification Rules (Mandatory)

- If a required value is missing or unclear, stop and ask a direct question before making assumptions.
- Do not silently choose `provider`, `model`, `ui`, or `modules.tools` when the user did not specify them.
- Always ask for `provider`, `model`, and `ui` explicitly.
- Never switch to creating files under `SKILLS/` unless the user explicitly changes the request to "create a skill".
- Apply this skill as the source of truth for agent creation/editing; do not improvise alternate agent formats.
- Ask in this strict order:
  1. `provider`
  2. `model` (from the chosen provider)
  3. `ui`
- When asking about `runtime.provider`, include the available providers list from `AIDZeroCode/LLMProviders/`.
- After the user chooses `provider`, list only models available for that provider and ask the user to choose one.
- When asking about `runtime.ui`, include a short list of currently available UIs found in `AIDZeroCode/UI/`.
- When asking about `modules.tools`, include a short list of currently available tools found in `AIDZeroCode/TOOLS/` (tool module names or `TOOL_NAME` values).
- If multiple unclear fields exist, ask in one compact message with separate lines per field.

Discovery commands (before asking):

```bash
uv run python -c "from pathlib import Path; import AIDZero as L; print('\\n'.join(L._discover_providers(Path('.'))))"
uv run python -c "from pathlib import Path; import AIDZero as L; provider='openai'; print('\\n'.join(L._list_provider_models(Path('.'), provider)))"
ls -1 AIDZeroCode/UI
ls -1 AIDZeroCode/TOOLS
```

Example prompts to the user:
- `Necesito que elijas proveedor para este agente. Opciones disponibles: openai, claude, google_gemini, lmstudio, ollama, llamacpp. Cual queres usar?`
- `Para el proveedor openai, estos son los modelos disponibles: gpt-4o-mini, gpt-4.1, ... Cual queres usar?`
- `No me quedo claro que UI queres para este agente. Opciones disponibles: tui, AndroidApp, Whatsapp. Cual queres usar?`
- `No me quedo claro que tools queres habilitar. Disponibles: read_text, write_text, list_files, sandbox_run, ... Queres 'all' o una lista especifica?`

## Prompt File Rules

- Put the agent behavior in `system_prompt.md`.
- Keep tool-calling instructions aligned with current runtime conventions.
- Do not reference files outside `Agents/`; profile prompt paths are constrained to stay inside that tree.

## Headless Rules

- Headless mode uses `Agents/default/HeadlessPrompt.txt` when launching `uv run AIDZero.py --headless`.
- If the task changes headless behavior, update both the agent folder contents and any affected tests.

## Validation

Run the smallest relevant checks after changes:

```bash
uv run --with pytest pytest AIDZeroCode/tests/test_agents.py -q
uv run --with pytest pytest AIDZeroCode/tests/test_root_launcher_headless.py -q
```

If you also changed API/profile serialization or slash command messaging, run the related targeted tests too.

## Output Expectations

When finishing work with this skill, report:

1. which agent folders/files were changed
2. the resulting runtime behavior
3. validation commands run and whether they passed
4. any remaining compatibility caveats
