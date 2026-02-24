#!/usr/bin/env python3
"""Terminal interface for the AgentCreator core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.provider_registry import ProviderRegistry
from agent.service import AgentCreator
from agent.ui_display import to_ui_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new agent project from a user request.")
    parser.add_argument(
        "--request",
        help="Natural-language description of the agent to create. If omitted, interactive prompt is used.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Model name for the selected provider (default: %(default)s).",
    )
    parser.add_argument(
        "--provider",
        default="AID-google_gemini",
        help="Provider name (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate and print the plan. Do not scaffold files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow scaffolding into a non-empty destination.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation and scaffold immediately (unless --dry-run).",
    )
    return parser.parse_args()


def run_terminal_agent(
    *,
    provider_name: str,
    model: str,
    user_request: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    yes: bool = False,
    repo_root: Path | None = None,
) -> int:
    provider_registry = ProviderRegistry(repo_root or Path.cwd())
    selected_provider_name = provider_name.strip()
    selected_model_name = model.strip()
    root = (repo_root or Path.cwd()).resolve()

    request = (user_request or "").strip()
    if not request:
        request = input("Describe the agent you want to create:\n> ").strip()
    if not request:
        print("error> empty request.")
        return 2

    try:
        provider = provider_registry.create(selected_provider_name)
    except Exception as error:  # noqa: BLE001
        print(f"error> could not initialize provider '{to_ui_label(selected_provider_name)}': {error}")
        return 2

    creator = AgentCreator(provider=provider, model=selected_model_name, repo_root=root)
    try:
        planning_result = creator.describe_requirements(user_request=request)
    except Exception as error:  # noqa: BLE001
        print(f"error> planning failed: {error}")
        return 1

    print("\n=== Agent Plan ===")
    print(json.dumps(planning_result.plan.to_dict(), indent=2, ensure_ascii=False))
    print(f"\nPlanned project folder: {planning_result.plan.project_folder}")

    if dry_run:
        return 0

    if not yes:
        confirmation = input("\nProceed with scaffolding? [y/N] ").strip().lower()
        if confirmation not in {"y", "yes"}:
            print("Scaffolding cancelled.")
            return 0

    try:
        scaffold_result = creator.create_agent_project_from_plan(
            user_request=request,
            plan=planning_result.plan,
            catalog=planning_result.catalog,
            overwrite=overwrite,
        )
    except Exception as error:  # noqa: BLE001
        print(f"error> scaffolding failed: {error}")
        return 1

    print("\n=== Scaffold Result ===")
    print(f"Destination: {scaffold_result.destination}")
    print(f"Created directories: {len(scaffold_result.created_directories)}")
    print(f"Copied items: {len(scaffold_result.copied_items)}")
    if scaffold_result.entrypoint_file:
        print(f"Entrypoint: {scaffold_result.entrypoint_file}")
    if scaffold_result.runtime_config_file:
        print(f"Runtime config: {scaffold_result.runtime_config_file}")
    if scaffold_result.metadata_file:
        print(f"Plan file: {scaffold_result.metadata_file}")
    return 0


def main() -> int:
    args = parse_args()
    return run_terminal_agent(
        provider_name=args.provider,
        model=args.model,
        user_request=args.request,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        yes=args.yes,
        repo_root=Path.cwd(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
