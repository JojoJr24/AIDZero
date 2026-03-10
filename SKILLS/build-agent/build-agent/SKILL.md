---
name: build-agent
description: Deprecated legacy skill. Do not use for current AIDZero runtime agent creation; use agent-creator instead.
---

# Build Agent (Deprecated)

## Status

This skill is deprecated and kept only for historical compatibility.

## Do Not Use For New Work

- Do not create `AGENTS/*.yaml` files.
- Do not create legacy `agent.yaml` profiles.
- Current runtime agent profiles must be created under `Agents/<name>/` using `agent-creator`.

## Migration Rule

If a request asks for a new runtime agent:

1. Use `agent-creator`.
2. Create/update:
   - `Agents/<name>/<name>.json`
   - `Agents/<name>/system_prompt.md`
   - `Agents/<name>/HeadlessPrompt.txt`
3. Ask for `provider`, then `model`, then `ui`, each with available options.
