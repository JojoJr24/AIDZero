#!/usr/bin/env python3
"""Root runtime launcher for the new OpenClaw-like project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from agent.models import RuntimeConfig
from agent.runtime_config import RuntimeConfigStore
from agent.ui_registry import UIRegistry


@dataclass(frozen=True)
class CliArgs:
    request: str | None
    ui: str | None
    provider: str | None
    model: str | None
    trigger: str
    reconfigure: bool
    list_options: bool
    ui_options: list[str]


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="AIDZero runtime launcher.")
    parser.add_argument("--request", help="Prompt to process.")
    parser.add_argument("--ui", default=None, help="UI runtime (default: terminal).")
    parser.add_argument("--provider", default=None, help="Provider folder in LLMProviders/.")
    parser.add_argument("--model", default=None, help="Model name.")
    parser.add_argument(
        "--trigger",
        default="interactive",
        choices=["interactive", "heartbeat", "cron", "messengers", "webhooks", "all"],
        help="Gateway trigger source to consume.",
    )
    parser.add_argument("--reconfigure", action="store_true", help="Ask and persist runtime choices.")
    parser.add_argument("--list-options", action="store_true", help="List UI and provider options.")
    parser.add_argument(
        "--ui-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="UI option passed through to selected UI.",
    )
    parsed = parser.parse_args()
    return CliArgs(
        request=parsed.request,
        ui=parsed.ui,
        provider=parsed.provider,
        model=parsed.model,
        trigger=parsed.trigger,
        reconfigure=parsed.reconfigure,
        list_options=parsed.list_options,
        ui_options=parsed.ui_option,
    )


def _discover_providers(repo_root: Path) -> list[str]:
    root = repo_root / "LLMProviders"
    if not root.exists():
        return []
    names: list[str] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not entry.is_dir():
            continue
        if (entry / "provider.py").is_file():
            names.append(entry.name)
    return names


def _parse_ui_options(raw_items: list[str], trigger: str) -> dict[str, str]:
    options: dict[str, str] = {"trigger": trigger}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --ui-option '{item}'. Use KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --ui-option '{item}'. KEY cannot be empty.")
        options[key] = value.strip()
    return options


def _pick(prompt: str, options: list[str], default: str | None = None) -> str:
    if not options:
        raise RuntimeError(f"No options available for: {prompt}")

    normalized_default = default if default in options else options[0]
    print(f"\n{prompt}")
    for index, option in enumerate(options, start=1):
        marker = " (default)" if option == normalized_default else ""
        print(f"{index}. {option}{marker}")

    while True:
        raw = input(f"Choose 1-{len(options)} (Enter for default): ").strip()
        if not raw:
            return normalized_default
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Invalid selection.")


def _ensure_runtime_config(
    *,
    repo_root: Path,
    store: RuntimeConfigStore,
    ui_options: list[str],
    force_reconfigure: bool,
) -> tuple[RuntimeConfig, dict[str, str]]:
    ui_registry = UIRegistry(repo_root)
    ui_names = ui_registry.names()
    provider_names = _discover_providers(repo_root)

    if not ui_names:
        raise RuntimeError("No runnable UI found under UI/<name>/entrypoint.py")
    if not provider_names:
        raise RuntimeError("No providers found under LLMProviders/<name>/provider.py")

    loaded = None if force_reconfigure else store.load()
    if loaded and loaded.ui in ui_names and loaded.provider in provider_names:
        options = _parse_ui_options(ui_options, trigger="interactive")
        return loaded, options

    ui = _pick("Select UI:", ui_names, default="terminal")
    provider = _pick("Select provider:", provider_names)
    model = input("Model name (required): ").strip()
    if not model:
        raise RuntimeError("Model name cannot be empty.")

    config = RuntimeConfig(ui=ui, provider=provider, model=model)
    store.save(config)
    options = _parse_ui_options(ui_options, trigger="interactive")
    return config, options


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent

    ui_registry = UIRegistry(repo_root)
    ui_names = ui_registry.names()
    provider_names = _discover_providers(repo_root)

    if args.list_options:
        print("UI options:")
        for ui_name in ui_names:
            print(f"- {ui_name}")
        print("Provider options:")
        for provider_name in provider_names:
            print(f"- {provider_name}")
        return 0

    store = RuntimeConfigStore(repo_root)

    if args.ui and args.provider and args.model:
        config = RuntimeConfig(ui=args.ui.strip(), provider=args.provider.strip(), model=args.model.strip())
        store.save(config)
    else:
        config, _ = _ensure_runtime_config(
            repo_root=repo_root,
            store=store,
            ui_options=args.ui_options,
            force_reconfigure=args.reconfigure,
        )

    if config.ui not in ui_names:
        print(f"error> unknown ui '{config.ui}'")
        return 2
    if config.provider not in provider_names:
        print(f"error> unknown provider '{config.provider}'")
        return 2

    try:
        parsed_ui_options = _parse_ui_options(args.ui_options, args.trigger)
    except ValueError as error:
        print(f"error> {error}")
        return 2

    print("Active runtime:")
    print(f"- ui: {config.ui}")
    print(f"- provider: {config.provider}")
    print(f"- model: {config.model}")
    print(f"- trigger: {args.trigger}")

    return ui_registry.run(
        config.ui,
        provider_name=config.provider,
        model=config.model,
        user_request=args.request,
        repo_root=repo_root,
        ui_options=parsed_ui_options,
    )


if __name__ == "__main__":
    raise SystemExit(main())
