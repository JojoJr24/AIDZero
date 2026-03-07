"""Dedicated launcher for UI layer connected to a core API."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from CORE.api_client import CoreAPIError
from CORE.agents import AgentProfileManager
from CORE.ui_registry import UIRegistry


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


def _parse_args() -> Any:
    parser = ArgumentParser(description="Run only the UI layer against a core API")
    parser.add_argument("--ui", default=None, help="UI name from UI/<name>/entrypoint.py")
    parser.add_argument("--core-url", default="http://127.0.0.1:8765", help="Core API URL")
    parser.add_argument("--request", help="Optional one-shot prompt")
    parser.add_argument("--agent", default=None, help="Agent profile name from Agents/<name>/<name>.json")
    parser.add_argument(
        "--ui-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra option passed to UI entrypoint (repeatable)",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root path",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()

    profile_manager = AgentProfileManager(repo_root)
    if args.agent and str(args.agent).strip():
        try:
            active_profile = profile_manager.set_active_profile(str(args.agent).strip())
        except Exception as error:  # noqa: BLE001
            print(f"error> {error}")
            return 2
    else:
        active_profile = profile_manager.get_active_profile()

    ui_registry = UIRegistry(repo_root)
    ui_name = (str(args.ui).strip() if args.ui else active_profile.runtime_ui)
    if ui_name not in ui_registry.names():
        print(f"error> unknown ui '{ui_name}'")
        return 2
    if ui_registry.ui_type(ui_name) == "thirdparty":
        print(f"error> ui '{ui_name}' is thirdparty and has no local launcher.")
        print("tip> Ejecutá aidzero-core y conectá el cliente externo por LAN.")
        return 2

    try:
        ui_options = _parse_ui_options(list(args.ui_option), trigger="interactive")
    except ValueError as error:
        print(f"error> {error}")
        return 2
    ui_options["core_url"] = str(args.core_url).strip()

    print("UI runtime (remote core):")
    print(f"- ui: {ui_name}")
    print(f"- core_url: {ui_options['core_url']}")
    print(f"- agent: {active_profile.name}")

    try:
        return ui_registry.run(
            ui_name,
            provider_name=active_profile.runtime_provider,
            model=active_profile.runtime_model,
            user_request=args.request,
            repo_root=repo_root,
            ui_options=ui_options,
        )
    except CoreAPIError as error:
        print(f"error> cannot reach core API at {ui_options['core_url']}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
