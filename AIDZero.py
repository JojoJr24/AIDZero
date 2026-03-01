#!/usr/bin/env python3
"""Root entrypoint for AIDZero runtime orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.catalog import ComponentCatalogBuilder
from agent.models import ComponentCatalog
from agent.provider_registry import ProviderRegistry
from agent.runtime_config import RuntimeConfig, RuntimeConfigStore, ensure_runtime_config
from agent.ui_display import to_ui_label, to_ui_model_label
from agent.ui_registry import UIRegistry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIDZero root entrypoint.")
    parser.add_argument(
        "--request",
        help="Natural-language request to send to the selected UI.",
    )
    parser.add_argument(
        "--ui",
        default=None,
        help="UI runtime to execute (for example: terminal).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider to execute (for example: AID-google_gemini).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for the selected provider.",
    )
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Force interactive runtime setup and rewrite saved runtime config.",
    )
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="Print available UI/provider options and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and review a plan without scaffolding files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow scaffolding into a non-empty destination.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive review and continue with defaults.",
    )
    parser.add_argument(
        "--ui-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="UI-specific option key/value. Repeat as needed.",
    )
    return parser


def _parse_ui_options(raw_values: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for raw in raw_values:
        value = raw.strip()
        if not value:
            continue
        if "=" not in value:
            raise ValueError(f"Invalid --ui-option '{raw}'. Use KEY=VALUE format.")
        key, parsed_value = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --ui-option '{raw}'. KEY cannot be empty.")
        options[key] = parsed_value.strip()
    return options


def _runtime_catalog(catalog: ComponentCatalog, ui_names: set[str]) -> ComponentCatalog:
    return ComponentCatalog(
        root=catalog.root,
        llm_providers=list(catalog.llm_providers),
        skills=list(catalog.skills),
        tools=list(catalog.tools),
        mcp=list(catalog.mcp),
        ui=[item for item in catalog.ui if item.name in ui_names],
    )


def _print_available_options(*, ui_names: list[str], provider_names: list[str]) -> None:
    print("Available UI options:")
    for ui_name in ui_names:
        print(f"- {to_ui_label(ui_name)}")
    print("Available provider options:")
    for provider_name in provider_names:
        print(f"- {to_ui_label(provider_name)}")


def _save_runtime_override(
    *,
    store: RuntimeConfigStore,
    ui: str,
    provider: str,
    model: str,
    default_generation_log_enabled: bool,
) -> None:
    existing = store.load()
    generation_log_enabled = (
        existing.generation_process_log_enabled
        if existing is not None
        else default_generation_log_enabled
    )
    store.save(
        RuntimeConfig(
            ui=ui,
            provider=provider,
            model=model,
            generation_process_log_enabled=generation_log_enabled,
        )
    )


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent

    catalog = ComponentCatalogBuilder(repo_root).build()
    provider_registry = ProviderRegistry(repo_root)
    ui_registry = UIRegistry(repo_root)

    ui_names = ui_registry.names()
    provider_names = provider_registry.names()
    ui_name_set = set(ui_names)

    if not ui_names:
        print("error> no runnable UI found in UI/*/entrypoint.py")
        return 2

    if args.list_options:
        _print_available_options(ui_names=ui_names, provider_names=provider_names)
        return 0

    config_store = RuntimeConfigStore(repo_root)
    runtime_config = ensure_runtime_config(
        catalog=_runtime_catalog(catalog, ui_name_set),
        provider_registry=provider_registry,
        store=config_store,
        reconfigure=args.reconfigure,
    )

    selected_ui = (args.ui or runtime_config.ui).strip()
    selected_provider = (args.provider or runtime_config.provider).strip()
    selected_model = (args.model or runtime_config.model).strip()
    if not selected_model:
        selected_model = provider_registry.default_model(selected_provider)

    if selected_ui not in ui_name_set:
        print(f"error> unknown ui '{to_ui_label(selected_ui)}'.")
        print(f"Use one of: {', '.join(to_ui_label(item) for item in ui_names)}")
        return 2

    if not provider_registry.has(selected_provider):
        print(f"error> unknown provider '{to_ui_label(selected_provider)}'.")
        print(f"Use one of: {', '.join(to_ui_label(item) for item in provider_names)}")
        return 2

    if args.ui or args.provider or args.model:
        _save_runtime_override(
            store=config_store,
            ui=selected_ui,
            provider=selected_provider,
            model=selected_model,
            default_generation_log_enabled=runtime_config.generation_process_log_enabled,
        )

    try:
        ui_options = _parse_ui_options(args.ui_option)
    except ValueError as error:
        print(f"error> {error}")
        return 2

    print(
        "Active runtime configuration:\n"
        f"- ui: {to_ui_label(selected_ui)}\n"
        f"- provider: {to_ui_label(selected_provider)}\n"
        f"- model: {to_ui_model_label(selected_model)}\n"
        f"- generation process log: "
        f"{'enabled' if runtime_config.generation_process_log_enabled else 'disabled'}"
    )

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
