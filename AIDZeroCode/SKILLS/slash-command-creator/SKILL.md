---
name: slash-command-creator
description: Add or modify slash commands in the TUI, separating internal core commands from plugin-style commands implemented as self-contained files under DASH/.
---

# Slash Command Authoring

## Purpose

Implement reliable `/command` UX in this codebase:

- clear split between command types:
  - internal commands (core behavior in app/runtime code)
  - DASH plugin commands (self-contained command modules in `DASH/`)
- command discoverability via selector/autocomplete
- command execution wiring in the TUI handler
- consistent validation and regression checks
- zero-touch extensibility: new commands are added by creating one file under `DASH/`

## Trigger Conditions

Use this skill when the user asks to:

- add a new slash command
- change behavior of an existing slash command
- update command suggestions/autocomplete in the TUI
- adjust parsing/routing rules for `/...` input

Do not use this skill for purely internal commands that must stay in core files. In that case, apply direct code edits in the relevant core module.

## Execution Steps

1. Classify the requested command:
   - internal command: implemented in core code, outside this skill's plugin contract
   - DASH plugin command: implement with one self-contained `DASH/*.py` file
2. Inspect current command flow in `UI/tui/textual_app.py` (`on_input_submitted`, `on_input_changed`, `_handle_command`, selector helpers).
3. Inspect dynamic loader in `core/dash_commands.py` and current modules in `DASH/`.
4. Define the command contract first:
   - canonical command syntax (`/name`, `/name arg`)
   - aliases (if any)
   - side effect (state update, output line, exit, etc.)
5. For DASH plugin commands, add a new `DASH/<command_name>.py` module with this minimum contract:
   - `DASH_COMMANDS = [{"command": "/x", "description": "..."}, ...]`
   - `def match(raw: str) -> bool`
   - `def run(raw: str, *, app) -> bool | str | None`
   - optional `DASH_PRIORITY = <int>` for dispatch order
6. Keep DASH plugin code self-contained in that file:
   - parse args in-module
   - implement side effects in-module
   - avoid spreading command-specific logic into `UI/` or `core/`
   - if the command should produce conversational text, return `str` (that value is written into the input)
7. Keep selector behavior aligned with execution:
   - filter by prefix
   - completion on `Tab` and `Enter` when applicable
   - hide selector when input is not a slash command
8. Do not modify core code to register DASH commands. The loader discovers files automatically from `DASH/`.
9. Validate:
   - run targeted tests first (`uv run pytest -q tests/test_ui_registry.py tests/test_engine.py`)
   - run dash command tests (`uv run pytest -q tests/test_dash_commands.py`)
   - compile check for edited module (`uv run python -m py_compile UI/tui/textual_app.py`)
   - if full suite fails, report unrelated failures separately.

## Output Expectations

When completing a request with this skill, return:

1. list of changed files
2. brief behavior summary (what new `/command` does)
3. validation commands run and outcomes
4. any known limitations or follow-up actions

## Resources

- `scripts/`: executable helpers
- `references/`: load [overview.md](references/overview.md) for concrete command design patterns in this repository
- `assets/`: files meant for output usage, not context loading
