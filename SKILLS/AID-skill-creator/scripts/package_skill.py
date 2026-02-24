#!/usr/bin/env python3
"""Package a skill folder as a .skill archive."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from quick_validate import validate_skill


def collect_files(skill_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if path.name.startswith(".DS_Store"):
            continue
        if path.name.endswith("~"):
            continue
        files.append(path)
    return files


def package_skill(skill_path: Path, output_dir: Path | None = None) -> Path:
    skill_dir = skill_path.resolve()
    if not skill_dir.is_dir():
        raise NotADirectoryError(f"Not a skill directory: {skill_dir}")
    if not (skill_dir / "SKILL.md").exists():
        raise FileNotFoundError(f"SKILL.md missing in: {skill_dir}")

    valid, message = validate_skill(skill_dir)
    if not valid:
        raise ValueError(f"Validation failed: {message}")

    destination_root = output_dir.resolve() if output_dir else Path.cwd()
    destination_root.mkdir(parents=True, exist_ok=True)
    archive_path = destination_root / f"{skill_dir.name}.skill"

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in collect_files(skill_dir):
            arcname = file_path.relative_to(skill_dir.parent)
            archive.write(file_path, arcname)
            print(f"added: {arcname}")

    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a .skill archive from a skill folder.")
    parser.add_argument("skill_path", help="Path to the skill folder.")
    parser.add_argument("output_dir", nargs="?", help="Optional output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    skill_path = Path(args.skill_path)
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        archive_path = package_skill(skill_path=skill_path, output_dir=output_dir)
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}")
        return 1

    print(f"Created archive: {archive_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
