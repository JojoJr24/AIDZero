#!/usr/bin/env python3
"""Initialize a new skill directory with starter files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NAME_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")

SKILL_MD_TEMPLATE = """---
name: {skill_name}
description: [TODO: Explain what this skill does and, most importantly, when it should be used.]
---

# {skill_title}

## Purpose
[TODO: Explain the outcome this skill enables.]

## Trigger Conditions
[TODO: Describe concrete user requests, contexts, file types, or signals that should activate this skill.]

## Execution Steps
1. [TODO: First operation]
2. [TODO: Next operation]
3. [TODO: Final operation]

## Output Expectations
[TODO: Define expected structure/style for responses produced with this skill.]

## Resources
- `scripts/`: executable helpers
- `references/`: deep documentation to load when needed
- `assets/`: files meant for output usage, not context loading
"""

EXAMPLE_SCRIPT = """#!/usr/bin/env python3
\"\"\"Example helper for {skill_name}.\"\"\"

def main() -> int:
    print("Replace this with real automation logic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

EXAMPLE_REFERENCE = """# {skill_title} Reference

Use this file for detail that should not clutter `SKILL.md`.

## Suggested Content
- Domain-specific constraints
- Extended examples
- Error-handling patterns
- Integration notes
"""

EXAMPLE_ASSET = """This placeholder file represents assets bundled with the skill.
Replace with templates, media, starter projects, or remove it if unnecessary.
"""


def title_case(skill_name: str) -> str:
    return " ".join(chunk.capitalize() for chunk in skill_name.split("-"))


def validate_name(skill_name: str) -> str | None:
    if len(skill_name) > 64:
        return "Skill name must be 64 characters or fewer."
    if not NAME_PATTERN.fullmatch(skill_name):
        return "Skill name must use letters, digits, and hyphens only."
    if skill_name.startswith("-") or skill_name.endswith("-") or "--" in skill_name:
        return "Skill name cannot start/end with hyphen or contain consecutive hyphens."
    return None


def write_file(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def init_skill(skill_name: str, root_path: Path) -> Path:
    skill_dir = root_path.resolve() / skill_name
    if skill_dir.exists():
        raise FileExistsError(f"Skill directory already exists: {skill_dir}")

    skill_dir.mkdir(parents=True, exist_ok=False)
    skill_title = title_case(skill_name)

    write_file(
        skill_dir / "SKILL.md",
        SKILL_MD_TEMPLATE.format(skill_name=skill_name, skill_title=skill_title),
    )
    write_file(
        skill_dir / "scripts" / "example.py",
        EXAMPLE_SCRIPT.format(skill_name=skill_name),
        executable=True,
    )
    write_file(
        skill_dir / "references" / "overview.md",
        EXAMPLE_REFERENCE.format(skill_title=skill_title),
    )
    write_file(skill_dir / "assets" / "README.txt", EXAMPLE_ASSET)
    return skill_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a skill folder with starter SKILL.md and resources."
    )
    parser.add_argument("skill_name", help="Skill identifier (hyphen-case).")
    parser.add_argument("--path", required=True, help="Parent directory where the skill will be created.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation_error = validate_name(args.skill_name.strip())
    if validation_error:
        print(f"Error: {validation_error}")
        return 1

    try:
        created = init_skill(args.skill_name.strip(), Path(args.path))
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}")
        return 1

    print(f"Created skill at: {created}")
    print("Next steps:")
    print("1. Complete TODOs in SKILL.md.")
    print("2. Keep or remove starter files in scripts/references/assets.")
    print("3. Run: scripts/quick_validate.py <path/to/skill>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
