# UI Runtime Contract (v0.4.0)

## Purpose

- This folder contains runnable UI implementations only.
- The core runtime contract is stable enough that new UIs should be added as isolated folders under `UI/` instead of branching existing launchers.

## Required Folder Layout

- Place each runnable UI in `UI/<Name>/`.
- The entrypoint must be `UI/<Name>/entrypoint.py`.
- Keep UI-specific helpers in the same folder when they are not shared by other UIs.
- Do not move UI-only concerns into `CORE/` unless the code becomes reusable across multiple interfaces.

## Required Entry Point

- Every runnable UI must export `run_ui(...)`.
- `run_ui(...)` must return an integer exit code.
- Keep the entrypoint signature aligned with the existing runtime launchers:

```python
def run_ui(
    *,
    provider_name: str,
    model: str,
    user_request: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    yes: bool = False,
    repo_root: Path | None = None,
    ui_options: dict[str, str] | None = None,
) -> int:
    ...
```

## Discovery Rules

- Discovery is dynamic through `CORE/ui_registry.py`.
- Any folder under `UI/` with an `entrypoint.py` becomes selectable through `--ui <Name>`.
- The folder name becomes CLI surface. Choose it intentionally and document the exact casing.

## Runtime Wiring

- For embedded/local core mode, construct runtime state with `CORE.ui_runtime.build_ui_runtime(...)`.
- For remote-core mode, read `core_url` from `ui_options` and use `CORE.api_client` proxies.
- A UI that can target a remote core should verify connectivity at startup when practical.
- Reuse the active agent profile from the runtime or remote core instead of loading raw profile files again.

## Event, History, and Session Rules

- UIs are responsible for turning input into `CORE.models.TriggerEvent`.
- Use `kind="interactive"` unless the feature explicitly introduces a new end-to-end trigger type.
- Set `source` to the actual channel name, not `terminal` by default.
- Preserve prompt history by calling `history.add_prompt(prompt)` in the UI layer, matching the TUI behavior.
- If the UI supports ongoing chats, expose a way to reset the current session through `engine.reset_session()` when available.
- If the UI only supports a single session or a single sender, document that explicitly in the module and tests.

## UI Options

- UI-specific settings must be passed only through repeated `--ui-option KEY=VALUE`.
- Parse and validate `ui_options` inside the UI entrypoint.
- Invalid UI options should fail cleanly without traceback spam when possible.
- Prefer transport-specific names such as `webhook_port`, `webhook_path`, `response_format`.

## Networked UI Guidance

- If the UI exposes HTTP endpoints, add a lightweight health endpoint.
- Keep webhook or bot handlers thin: parse request, build event metadata, call the engine, format the reply.
- Prefer standard-library HTTP handling unless the UI clearly needs a larger framework.
- Return user-visible transport responses on runtime failures so the user sees the error instead of a silent timeout.
- Be explicit about the inbound payload shapes and transport expectations the UI supports.

## Current Supported UI

- `tui` via `UI/tui/entrypoint.py` and `UI/tui/textual_app.py`
- `Whatsapp` via `UI/Whatsapp/entrypoint.py`

## Testing Expectations

- Add focused tests for every new UI behavior change.
- At minimum, cover:
  - entrypoint option parsing
  - prompt/history integration
  - event creation and source metadata
  - remote-core mode when applicable
  - transport-specific request/response formatting
- Keep `tests/test_ui_registry.py` green by preserving the `UI/<Name>/entrypoint.py` contract.

## Validation

```bash
uv run --with pytest pytest tests/test_ui_registry.py tests/test_ui_launcher.py tests/test_whatsapp_ui.py tests/test_tui_textual_app.py -q
```
