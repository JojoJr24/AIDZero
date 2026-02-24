#!/usr/bin/env python3
"""Root entrypoint for AIDZero runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_creator.catalog import ComponentCatalogBuilder
from agent_creator.provider_registry import ProviderRegistry
from agent_creator.runtime_config import RuntimeConfig, RuntimeConfigStore, ensure_runtime_config
from UI.terminal.agent_creator_terminal import run_terminal_agent_creator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIDZero root entrypoint.")
    parser.add_argument(
        "--request",
        help="Natural-language description of the agent to create. If omitted, interactive prompt is used.",
    )
    parser.add_argument(
        "--ui",
        default=None,
        help="UI to use for this run (example: terminal).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider to use for this run (example: AID-google_gemini).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use for this run.",
    )
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Force interactive startup setup and rewrite saved runtime configuration.",
    )
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List available UI and provider options, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and print plan without scaffolding.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow scaffolding into a non-empty destination.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip scaffolding confirmation prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    catalog = ComponentCatalogBuilder(repo_root).build()
    provider_registry = ProviderRegistry(repo_root)

    if args.list_options:
        print("Available UI options:")
        for item in catalog.ui:
            print(f"- {item.name}")
        print("Available provider options:")
        for provider_name in provider_registry.names():
            print(f"- {provider_name}")
        return 0

    config_store = RuntimeConfigStore(repo_root)
    runtime_config = ensure_runtime_config(
        catalog=catalog,
        provider_registry=provider_registry,
        store=config_store,
        reconfigure=args.reconfigure,
    )

    selected_ui = (args.ui or runtime_config.ui).strip()
    selected_provider = (args.provider or runtime_config.provider).strip()
    selected_model = (args.model or runtime_config.model).strip()
    if not selected_model:
        selected_model = provider_registry.default_model(selected_provider)

    available_ui = {item.name for item in catalog.ui}
    if available_ui and selected_ui not in available_ui:
        print(f"error> unknown ui '{selected_ui}'.")
        print(f"Use one of: {', '.join(sorted(available_ui))}")
        return 2

    if not provider_registry.has(selected_provider):
        print(f"error> unknown provider '{selected_provider}'.")
        print(f"Use one of: {', '.join(provider_registry.names())}")
        return 2

    if args.ui or args.provider or args.model:
        config_store.save(
            RuntimeConfig(
                ui=selected_ui,
                provider=selected_provider,
                model=selected_model,
            )
        )

    print(
        "Active runtime configuration:\n"
        f"- ui: {selected_ui}\n"
        f"- provider: {selected_provider}\n"
        f"- model: {selected_model}"
    )

    if selected_ui == "terminal":
        return run_terminal_agent_creator(
            provider_name=selected_provider,
            model=selected_model,
            user_request=args.request,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            yes=args.yes,
            repo_root=repo_root,
        )

    print(f"error> UI '{selected_ui}' is not implemented yet.")
    known_ui = [item.name for item in catalog.ui]
    if known_ui:
        print("Available UI entries in repository:")
        for ui_name in known_ui:
            print(f"- {ui_name}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
