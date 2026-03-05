# UI Runtime Contract (v0.2.0)

## Runnable UI Contract
- Runnable UIs must live in `UI/<name>/entrypoint.py`.
- Each runnable UI must export `run_ui(...)` and return an integer exit code.
- UI-specific settings must be passed through repeated `--ui-option KEY=VALUE`.

## Current UI Modules
- `UI/tui/entrypoint.py`: Textual UI entrypoint.
- `UI/tui/textual_app.py`: main Textual app behavior.

## Runtime Behavior Rules
- UIs must support interactive trigger handling.
- UIs must preserve prompt history behavior through the runtime history store.
- Remote-core mode must work when `core_url` is provided via `ui_options`.

## Validation
After UI changes:
1. `uv run AIDZero.py --request "ping"`
2. `uv run aidzero-ui --ui tui --core-url http://127.0.0.1:8765`
3. `pytest tests/test_ui_launcher.py tests/test_tui_textual_app.py -q`
