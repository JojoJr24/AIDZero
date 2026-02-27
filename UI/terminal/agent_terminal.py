#!/usr/bin/env python3
"""Terminal interface for the AgentCreator core."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.models import PlanningResult
from agent.provider_registry import ProviderRegistry
from agent.prompt_history import PromptHistoryStore
from agent.service import AgentCreator
from agent.ui_display import to_ui_label


def _format_list(values: list[str]) -> str:
    if not values:
        return "(none)"
    return ", ".join(values)


def _print_plan(plan_result: PlanningResult, *, revision_round: int) -> None:
    print("\n=== Agent Plan ===")
    print(f"Iteration: {revision_round + 1}")
    print(f"Agent: {plan_result.plan.agent_name}")
    print(f"Project folder: {plan_result.plan.project_folder}")
    print(f"Goal: {plan_result.plan.goal}")
    summary = plan_result.plan.summary.strip()
    if len(summary) > 420:
        summary = summary[:417].rstrip() + "..."
    print(f"Summary: {summary}")
    print("Requirements:")
    print(f"- Providers: {_format_list(plan_result.plan.required_llm_providers)}")
    print(f"- Skills: {_format_list(plan_result.plan.required_skills)}")
    print(f"- Tools: {_format_list(plan_result.plan.required_tools)}")
    print(f"- MCP: {_format_list(plan_result.plan.required_mcp)}")
    print(f"- UI: {_format_list(plan_result.plan.required_ui)}")
    if plan_result.plan.warnings:
        print(f"Warnings: {_format_list(plan_result.plan.warnings)}")


def _review_plan_interactively(
    *,
    creator: AgentCreator,
    request: str,
    initial_result: PlanningResult,
    dry_run: bool,
) -> PlanningResult | None:
    planning_result = initial_result
    revision_round = 0

    while True:
        _print_plan(planning_result, revision_round=revision_round)
        if dry_run:
            action = input(
                "\nPlan actions: [a]ccept and finish dry-run, [r]equest changes, [c]ancel\n> "
            ).strip().lower()
        else:
            action = input(
                "\nPlan actions: [a]ccept and continue to scaffold, [r]equest changes, [c]ancel\n> "
            ).strip().lower()

        if action in {"a", "accept"}:
            return planning_result
        if action in {"c", "cancel"}:
            print("Plan review cancelled.")
            return None
        if action not in {"r", "revise", "change"}:
            print("warning> invalid action. Use a, r, or c.")
            continue

        change_request = input("Describe the plan changes you want:\n> ").strip()
        if not change_request:
            print("warning> empty change request; plan not modified.")
            continue

        try:
            planning_result = creator.revise_requirements(
                user_request=request,
                current_plan=planning_result.plan,
                plan_change_request=change_request,
                ui_name="terminal",
            )
        except Exception as error:  # noqa: BLE001
            print(f"error> plan revision failed: {error}")
            continue
        revision_round += 1


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
        help="Skip interactive plan review and scaffold immediately (unless --dry-run).",
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
    prompt_history = PromptHistoryStore(root)

    request = (user_request or "").strip()
    if not request:
        request = _select_request_from_history(prompt_history)
    if not request:
        print("error> empty request.")
        return 2
    prompt_history.add_prompt(request)

    try:
        provider = provider_registry.create(selected_provider_name)
    except Exception as error:  # noqa: BLE001
        print(f"error> could not initialize provider '{to_ui_label(selected_provider_name)}': {error}")
        return 2

    creator = AgentCreator(provider=provider, model=selected_model_name, repo_root=root)
    try:
        planning_result = creator.describe_requirements(user_request=request, ui_name="terminal")
    except Exception as error:  # noqa: BLE001
        print(f"error> planning failed: {error}")
        return 1

    final_planning_result = planning_result
    if yes:
        _print_plan(planning_result, revision_round=0)
    else:
        reviewed_result = _review_plan_interactively(
            creator=creator,
            request=request,
            initial_result=planning_result,
            dry_run=dry_run,
        )
        if reviewed_result is None:
            return 0
        final_planning_result = reviewed_result

    if dry_run:
        print("Dry-run completed with the approved plan.")
        return 0

    try:
        scaffold_result = creator.create_agent_project_from_plan(
            user_request=request,
            plan=final_planning_result.plan,
            catalog=final_planning_result.catalog,
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
    if scaffold_result.process_log_file:
        print(f"Process log: {scaffold_result.process_log_file}")
    return 0


def _select_request_from_history(prompt_history: PromptHistoryStore) -> str:
    history_items = prompt_history.list_prompts(limit=10)
    if history_items:
        print("\n=== Prompt History ===")
        for index, item in enumerate(history_items, start=1):
            print(f"{index}. {item}")
        selection = input(
            "Select a prompt number from history or press Enter to write a new one:\n> "
        ).strip()
        if selection:
            if selection.isdigit():
                selected_index = int(selection)
                if 1 <= selected_index <= len(history_items):
                    return history_items[selected_index - 1]
            print("warning> invalid history selection; using manual input.")

    return input("Describe the agent you want to create:\n> ").strip()


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
