---
name: skill-creator
description: Create or modify a Codex skill under SKILLS/ with clear triggers, concise workflow instructions, and optional scripts/references/assets. Use only when the user explicitly asks to build or refactor a skill.
---

# Skill Creator

## Mission

Design or refactor skills so they are easy to trigger and easy to execute. Keep `SKILL.md` focused on operational steps; move deep detail into `references/`.

## Scope Boundary (Critical)

- This skill is only for `SKILLS/<name>/...`.
- Do not use this skill to create runtime agent profiles.
- If the request is about creating/modifying an agent in `Agents/<name>/`, use `agent-creator` instead.

## Skill Contract

- Frontmatter in `SKILL.md` must stay minimal and explicit:
  - `name`
  - `description`
- Supporting folders are optional and should be added only when useful:
  - `scripts/`
  - `references/`
  - `assets/`

## Standard Execution Flow

1. Confirm intent is truly "skill creation/refactor" and not "agent profile creation".
2. Collect concrete examples of when the skill should be used.
3. Decide which reusable artifacts are needed (`scripts`, `references`, `assets`).
4. Initialize or reorganize the skill folder.
5. Write a trigger-accurate `description` in frontmatter.
6. Write the body as a procedural execution guide with clear expected output.
7. Run `scripts/quick_validate.py` on the skill folder.

## Trigger Quality Rules

- Put all "when to use this skill" logic in frontmatter `description`.
- Keep the body centered on "how to execute this skill".
- Prefer specific verbs and contexts over generic wording.

## Web Research Rule

If the produced skill needs online research, explicitly instruct it to use Deep Research MCP tools.

## Available Commands

```bash
scripts/init_skill.py <skill-name> --path <output-dir>
scripts/quick_validate.py <path/to/skill-folder>
scripts/package_skill.py <path/to/skill-folder>
```

## Supporting Docs

- `references/workflows.md`
- `references/output-patterns.md`
