# UI Runtime Contract

- Runnable UIs must live in `UI/<name>.py`.
- Each runnable UI must export `run_ui(...)` and return an integer exit code.
- UIs must persist prompt history between runs.
- UIs must provide built-in prompt history selection.
- UI-specific settings must be passed through repeated `--ui-option KEY=VALUE`.
