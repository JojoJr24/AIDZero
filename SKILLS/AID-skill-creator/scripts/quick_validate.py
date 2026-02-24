#!/usr/bin/env python3
"""Validate basic structure and frontmatter rules for a skill folder."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    yaml = None

ALLOWED_PROPERTIES = {"name", "description", "license", "allowed-tools", "metadata"}
NAME_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")
FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def validate_skill(skill_path: str | Path) -> tuple[bool, str]:
    skill_dir = Path(skill_path)
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return False, "SKILL.md not found."

    content = skill_md.read_text(encoding="utf-8")
    frontmatter = extract_frontmatter(content)
    if frontmatter is None:
        return False, "Invalid or missing YAML frontmatter."

    parsed = parse_frontmatter(frontmatter)
    if not isinstance(parsed, dict):
        return False, "Frontmatter must be a dictionary."

    unknown_keys = set(parsed.keys()) - ALLOWED_PROPERTIES
    if unknown_keys:
        allowed = ", ".join(sorted(ALLOWED_PROPERTIES))
        unexpected = ", ".join(sorted(unknown_keys))
        return False, f"Unexpected frontmatter keys: {unexpected}. Allowed: {allowed}."

    if "name" not in parsed:
        return False, "Missing required key: name."
    if "description" not in parsed:
        return False, "Missing required key: description."

    return validate_name_and_description(parsed.get("name"), parsed.get("description"))


def extract_frontmatter(content: str) -> str | None:
    if not content.startswith("---"):
        return None
    match = FRONTMATTER_PATTERN.match(content)
    if match is None:
        return None
    return match.group(1)


def parse_frontmatter(frontmatter_text: str) -> dict[str, Any] | None:
    if yaml is not None:
        try:
            parsed = yaml.safe_load(frontmatter_text)
        except Exception:  # noqa: BLE001
            return None
        return parsed if isinstance(parsed, dict) else None
    return parse_frontmatter_without_yaml(frontmatter_text)


def parse_frontmatter_without_yaml(frontmatter_text: str) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for raw_line in frontmatter_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            return None
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            return None
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        result[key] = value
    return result


def validate_name_and_description(name: Any, description: Any) -> tuple[bool, str]:
    if not isinstance(name, str):
        return False, f"name must be string, got {type(name).__name__}."
    clean_name = name.strip()
    if not clean_name:
        return False, "name cannot be empty."
    if len(clean_name) > 64:
        return False, "name exceeds 64 characters."
    if NAME_PATTERN.fullmatch(clean_name) is None:
        return False, "name must be hyphen-case with letters/digits/hyphens."
    if clean_name.startswith("-") or clean_name.endswith("-") or "--" in clean_name:
        return False, "name cannot start/end with hyphen or contain '--'."

    if not isinstance(description, str):
        return False, f"description must be string, got {type(description).__name__}."
    clean_description = description.strip()
    if not clean_description:
        return False, "description cannot be empty."
    if len(clean_description) > 1024:
        return False, "description exceeds 1024 characters."
    if "<" in clean_description or ">" in clean_description:
        return False, "description cannot contain angle brackets."

    return True, "Skill is valid."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick validator for skill frontmatter/structure.")
    parser.add_argument("skill_directory", help="Path to skill directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    valid, message = validate_skill(args.skill_directory)
    print(message)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
