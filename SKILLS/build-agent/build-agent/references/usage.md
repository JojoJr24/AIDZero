# Build Agent Usage (Deprecated)

This skill is legacy-only.

Do not use legacy `AGENTS/*.yaml` profiles in this repository.

Use `agent-creator` for all runtime agent work and create:

- `Agents/<name>/<name>.json`
- `Agents/<name>/system_prompt.md`
- `Agents/<name>/HeadlessPrompt.txt`

Mandatory question flow for agent creation:

1. Ask provider and show available providers.
2. Ask model and show models for the selected provider.
3. Ask UI and show available UIs.
4. Ask tools selection (`all` or explicit list).
