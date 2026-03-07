# Slash Command Authoring Reference

This reference documents practical patterns for implementing slash commands in this repository.

## Command Types
- Internal commands:
  - Implemented in core code (`UI/`, `core/`, or runtime internals).
  - Use this only for platform-level behavior that cannot live in a plugin.
- DASH plugin commands:
  - Implemented as one self-contained file under `DASH/`.
  - This is the default path for feature commands.

## Primary Integration Points
- `UI/tui/textual_app.py`
- `core/dash_commands.py`
- `DASH/*.py`
- `on_input_submitted`: final command resolution and dispatch
- `_handle_command`: command execution and validation
- `on_input_changed` + selector helpers: discovery and autocomplete behavior

## Command Design Rules
1. Keep one canonical command string.
2. Only add aliases when there is clear user value.
3. Use explicit, short error messages for invalid args.
4. Side effects must be visible in TUI (`_append_system_line` or status update).
5. Do not execute regular prompts when a slash command was handled.

## Plugin Contract (`DASH/<name>.py`)
```python
DASH_COMMANDS = [
    {"command": "/example", "description": "Run example action"},
]

def match(raw: str) -> bool:
    return raw == "/example"

def run(raw: str, *, app) -> bool:
    del raw
    app._append_system_line("Example command executed.")
    return True
```

Run return semantics:
- `str`: write returned text into input box (for user to submit as prompt)
- `bool`: handled/not handled flag
- `None`: treated as handled

### Self-Contained Rule
- Keep all command-specific logic in the `DASH/<name>.py` file.
- Avoid spreading parsing/logic for that command into core app files.
- Core code should only provide generic registry/dispatch plumbing.

### `/time` Example (writes current time into input)
```python
from datetime import datetime
from textual.widgets import TextArea

DASH_COMMANDS = [{"command": "/time", "description": "Insert current time in input"}]

def match(raw: str) -> bool:
    return raw.strip() == "/time"

def run(raw: str, *, app) -> bool:
    del raw, app
    return datetime.now().strftime("%H:%M:%S")
```

## Autocomplete Pattern
- Suggestions are derived from all `DASH_COMMANDS` entries loaded from `DASH/*.py`.
- Filter remains `command.startswith(query)` for predictable behavior.
- Apply completion from highlighted suggestion on `Tab` and before submit.
- Clear/hide selector when input no longer starts with `/`.

## Zero-Touch Extension Rule
- To add a command, create one new Python file in `DASH/` with the contract above.
- Do not edit `UI/tui/textual_app.py` or hardcoded registries for new commands.
- Reloading/rerunning the app is enough for discovery.

## Validation Checklist
- New command appears in selector after typing `/`.
- `Tab` completes to the expected command.
- `Enter` on partial command resolves consistently.
- Valid command performs expected side effect.
- Invalid command/args produce clear feedback, no crash.
