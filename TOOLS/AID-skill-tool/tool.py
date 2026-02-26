"""Tool to discover and interact with Agent Skills stored under SKILLS/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def skills_root(repo_root: Path) -> Path:
    return repo_root / "SKILLS"


def list_skills(repo_root: Path) -> list[dict[str, Any]]:
    root = skills_root(repo_root)
    if not root.is_dir():
        return []

    skills: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.exists():
            continue
        skills.append(
            {
                "name": entry.name,
                "path": str(entry.relative_to(repo_root)),
                "description": read_skill_description(skill_md),
            }
        )
    return skills


def read_skill_description(skill_md: Path) -> str | None:
    for raw_line in skill_md.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line == "---":
            continue
        if line.startswith("description:"):
            value = line.split(":", 1)[1].strip()
            return value.strip('"').strip("'")
    return None


def resolve_skill_path(repo_root: Path, skill_name: str) -> Path:
    name = skill_name.strip()
    if not name:
        raise ValueError("skill_name cannot be empty.")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("skill_name must be a direct folder name under SKILLS/.")
    target = skills_root(repo_root) / name
    if not target.is_dir():
        raise FileNotFoundError(f"Skill not found: {name}")
    skill_md = target / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"Skill folder '{name}' does not contain SKILL.md")
    return target


def read_skill(repo_root: Path, skill_name: str) -> dict[str, Any]:
    skill_path = resolve_skill_path(repo_root, skill_name)
    skill_md = skill_path / "SKILL.md"
    references = sorted(
        str(path.relative_to(skill_path))
        for path in (skill_path / "references").glob("**/*")
        if path.is_file()
    ) if (skill_path / "references").is_dir() else []
    scripts = sorted(
        str(path.relative_to(skill_path))
        for path in (skill_path / "scripts").glob("**/*")
        if path.is_file()
    ) if (skill_path / "scripts").is_dir() else []
    return {
        "name": skill_path.name,
        "path": str(skill_path.relative_to(repo_root)),
        "description": read_skill_description(skill_md),
        "skill_markdown": skill_md.read_text(encoding="utf-8"),
        "references": references,
        "scripts": scripts,
    }


def run_skill_script(
    repo_root: Path,
    *,
    skill_name: str,
    script_relative_path: str,
    script_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    skill_path = resolve_skill_path(repo_root, skill_name)
    script_path = (skill_path / "scripts" / script_relative_path).resolve()
    scripts_root = (skill_path / "scripts").resolve()
    try:
        script_path.relative_to(scripts_root)
    except ValueError as error:
        raise ValueError("script_relative_path must stay inside the skill scripts/ folder.") from error
    if not script_path.exists() or not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_relative_path}")

    command = [sys.executable, str(script_path)]
    if script_args:
        command.extend(script_args)
    return subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interact with Agent Skills under SKILLS/.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List installed skills.")
    list_parser.add_argument("--json", action="store_true", help="Print JSON output.")

    show_parser = subparsers.add_parser("show", help="Show one skill and its files.")
    show_parser.add_argument("--skill", required=True, help="Skill folder name under SKILLS/.")
    show_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    show_parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Exclude full SKILL.md content from output.",
    )

    run_parser = subparsers.add_parser("run-script", help="Run a script from SKILLS/<skill>/scripts/.")
    run_parser.add_argument("--skill", required=True, help="Skill folder name under SKILLS/.")
    run_parser.add_argument("--script", required=True, help="Relative path inside scripts/.")
    run_parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to the script (use -- before args).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = repo_root_from_file()

    try:
        if args.command == "list":
            payload = {"skills": list_skills(repo_root)}
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                for item in payload["skills"]:
                    description = item["description"] or "no description"
                    print(f"{item['name']}: {description}")
            return 0

        if args.command == "show":
            payload = read_skill(repo_root, args.skill)
            if args.no_markdown:
                payload.pop("skill_markdown", None)
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print(f"Skill: {payload['name']}")
                print(f"Path: {payload['path']}")
                print(f"Description: {payload['description'] or 'no description'}")
                print(f"References: {len(payload['references'])}")
                print(f"Scripts: {len(payload['scripts'])}")
                if "skill_markdown" in payload:
                    print("\n--- SKILL.md ---")
                    print(payload["skill_markdown"])
            return 0

        if args.command == "run-script":
            script_args = list(args.script_args)
            if script_args and script_args[0] == "--":
                script_args = script_args[1:]
            result = run_skill_script(
                repo_root,
                skill_name=args.skill,
                script_relative_path=args.script,
                script_args=script_args,
            )
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            return int(result.returncode)

        parser.error(f"Unknown command: {args.command}")
        return 2
    except Exception as error:  # noqa: BLE001
        print(f"error> {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
