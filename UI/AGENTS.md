# UI Interface Guidelines

This file defines the minimum behavior expected from any runtime UI implementation.

## Core Contract

- Every runnable UI lives in `UI/<ui_name>/entrypoint.py`.
- `entrypoint.py` must export `run_ui(...)` and return an integer exit code.
- UI modules must parse and apply their own `ui_options`.
- UIs must not require root-level UI-specific flags beyond `--ui-option KEY=VALUE`.

## Usability Baseline

- The UI must allow creating an agent with at least these inputs:
  - `provider`
  - `model`
  - `request` (natural-language prompt)
  - `dry_run`
  - `overwrite`
- The UI must clearly show planning/scaffolding results and failure messages.
- Display labels for provider/UI/tool names must use shared normalization helpers.

## Prompt History (Mandatory)

- The UI must persist prompt history between runs.
- The UI must provide a built-in way to select a previous prompt from history.
- History must keep most-recent prompts first and avoid duplicate entries.

## Web-Specific Requirement

- If a `dry_run` result is available, the Web UI must offer an action to scaffold using that existing dry-run plan (without re-planning).
