#!/usr/bin/env python3
"""Root entrypoint for AIDZero runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.catalog import ComponentCatalogBuilder
from agent.models import ComponentCatalog
from agent.provider_registry import ProviderRegistry
from agent.runtime_config import RuntimeConfig, RuntimeConfigStore, ensure_runtime_config
from agent.ui_display import to_ui_label, to_ui_model_label
from agent.ui_registry import UIRegistry


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
    parser.add_argument(
        "--ui-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="UI-specific option passed to the selected UI. Repeat as needed.",
    )
    return parser.parse_args()


def _parse_ui_options(raw_options: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_item in raw_options:
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --ui-option '{raw_item}'. Use KEY=VALUE format.")
        key, value = item.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError(f"Invalid --ui-option '{raw_item}'. KEY cannot be empty.")
        parsed[normalized_key] = value.strip()
    return parsed


def _build_runtime_catalog(catalog: ComponentCatalog, available_ui_names: set[str]) -> ComponentCatalog:
    filtered_ui = [item for item in catalog.ui if item.name in available_ui_names]
    return ComponentCatalog(
        root=catalog.root,
        llm_providers=catalog.llm_providers,
        skills=catalog.skills,
        tools=catalog.tools,
        mcp=catalog.mcp,
        ui=filtered_ui,
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    catalog = ComponentCatalogBuilder(repo_root).build()
    provider_registry = ProviderRegistry(repo_root)
    ui_registry = UIRegistry(repo_root)
    available_ui = ui_registry.names()
    available_ui_set = set(available_ui)

    if not available_ui:
        print("error> no runnable UI found in UI/*/entrypoint.py")
        return 2

    if args.list_options:
        print("Available UI options:")
        for ui_name in available_ui:
            print(f"- {to_ui_label(ui_name)}")
        print("Available provider options:")
        for provider_name in provider_registry.names():
            print(f"- {to_ui_label(provider_name)}")
        return 0

    runtime_catalog = _build_runtime_catalog(catalog, available_ui_set)
    config_store = RuntimeConfigStore(repo_root)
    runtime_config = ensure_runtime_config(
        catalog=runtime_catalog,
        provider_registry=provider_registry,
        store=config_store,
        reconfigure=args.reconfigure,
    )

    selected_ui = (args.ui or runtime_config.ui).strip()
    selected_provider = (args.provider or runtime_config.provider).strip()
    selected_model = (args.model or runtime_config.model).strip()
    if not selected_model:
        selected_model = provider_registry.default_model(selected_provider)

    if selected_ui not in available_ui_set:
        print(f"error> unknown ui '{to_ui_label(selected_ui)}'.")
        print(f"Use one of: {', '.join(to_ui_label(item) for item in available_ui)}")
        return 2

    if not provider_registry.has(selected_provider):
        print(f"error> unknown provider '{to_ui_label(selected_provider)}'.")
        print(f"Use one of: {', '.join(to_ui_label(item) for item in provider_registry.names())}")
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
        f"- ui: {to_ui_label(selected_ui)}\n"
        f"- provider: {to_ui_label(selected_provider)}\n"
        f"- model: {to_ui_model_label(selected_model)}"
    )
    try:
        ui_options = _parse_ui_options(args.ui_option)
    except ValueError as error:
        print(f"error> {error}")
        return 2

    try:
        return ui_registry.run(
            selected_ui,
            provider_name=selected_provider,
            model=selected_model,
            user_request=args.request,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            yes=args.yes,
            repo_root=repo_root,
            ui_options=ui_options,
        )
    except Exception as error:  # noqa: BLE001
        print(f"error> UI '{to_ui_label(selected_ui)}' failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
