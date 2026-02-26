---
name: AID-skill-creator
description: Create or modify an Agent skill with clear triggers, concise workflow instructions, and optional scripts/references/assets. Use when the user asks to build a new skill or refactor an existing one.
---

# Skill Creator

## Mission

Design or refactor skills so they are easy to trigger and easy to execute. Keep `SKILL.md` focused on operational steps; move deep detail into `references/`.

## Skill Contract

- Frontmatter in `SKILL.md` must stay minimal and explicit:
  - `name`
  - `description`
- Supporting folders are optional and should be added only when useful:
  - `scripts/`
  - `references/`
  - `assets/`

## Standard Execution Flow

1. Collect concrete examples of when the skill should be used.
2. Decide which reusable artifacts are needed (`scripts`, `references`, `assets`).
3. Initialize or reorganize the skill folder.
4. Write a trigger-accurate `description` in frontmatter.
5. Write the body as a procedural execution guide with clear expected output.
6. Run `scripts/quick_validate.py` on the skill folder.

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
