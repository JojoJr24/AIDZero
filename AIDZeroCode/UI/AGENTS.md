# UI Runtime Contract (v0.3.0)

## Runnable UI Contract
- Runnable UIs must live in `UI/<name>/entrypoint.py`.
- Each runnable UI must export `run_ui(...)` and return an integer exit code.
- UI-specific settings must be passed through repeated `--ui-option KEY=VALUE`.

## Current Supported UI
- `tui` via `UI/tui/entrypoint.py` and `UI/tui/textual_app.py`.

## Runtime Behavior Rules
- Support interactive trigger handling.
- Preserve prompt history integration through runtime history store.
- Support remote-core mode when `core_url` is provided in `ui_options`.
- Keep streaming + artifact rendering stable under long responses.

## Validation
```bash
uv run --with pytest pytest tests/test_ui_launcher.py tests/test_tui_textual_app.py -q
```
