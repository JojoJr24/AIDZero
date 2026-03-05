#!/usr/bin/env python3
"""Root runtime launcher for the AIDZero agent runtime project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

from core.agents import AgentProfileManager
from core.models import RuntimeConfig
from core.ui_registry import UIRegistry


@dataclass(frozen=True)
class CliArgs:
    request: str | None
    agent: str | None


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="AIDZero runtime launcher.")
    parser.add_argument("--request", help="Prompt to process.")
    parser.add_argument("--agent", default=None, help="Agent profile name from Agents/*.json.")
    parsed = parser.parse_args()
    return CliArgs(
        request=parsed.request,
        agent=parsed.agent,
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


def _load_provider_module(repo_root: Path, provider_name: str) -> ModuleType:
    provider_file = repo_root / "LLMProviders" / provider_name / "provider.py"
    if not provider_file.is_file():
        raise FileNotFoundError(f"Provider file not found: {provider_file}")

    module_name = f"aidzero_provider_selector_{provider_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, provider_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load provider module: {provider_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_provider_class(module: ModuleType) -> type[object]:
    candidates: list[type[object]] = []
    for _, member in inspect.getmembers(module, inspect.isclass):
        if member.__module__ != module.__name__:
            continue
        if not member.__name__.endswith("Provider"):
            continue
        if member.__name__ == "OpenAICompatibleProvider":
            continue
        candidates.append(member)

    if not candidates:
        raise RuntimeError(f"No provider class found in '{module.__name__}'.")
    candidates.sort(key=lambda item: item.__name__)
    return candidates[0]


def _list_provider_models(repo_root: Path, provider_name: str) -> list[str]:
    module = _load_provider_module(repo_root, provider_name)
    provider_cls = _find_provider_class(module)
    provider = provider_cls()

    model_names: list[str] = []
    if hasattr(provider, "list_model_names"):
        try:
            raw_names = provider.list_model_names()
            if isinstance(raw_names, list):
                for item in raw_names:
                    if isinstance(item, str) and item.strip():
                        model_names.append(item.strip())
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"Could not list models using list_model_names(): {error}") from error
    elif hasattr(provider, "list_models"):
        try:
            raw_models = provider.list_models()
            if isinstance(raw_models, list):
                for item in raw_models:
                    if not isinstance(item, dict):
                        continue
                    for key in ("id", "name", "model"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            model_names.append(value.strip())
                            break
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"Could not list models using list_models(): {error}") from error

    unique: list[str] = []
    for model_name in model_names:
        if model_name not in unique:
            unique.append(model_name)
    return unique


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent

    ui_registry = UIRegistry(repo_root)
    ui_names = ui_registry.names()
    provider_names = _discover_providers(repo_root)

    profile_manager = AgentProfileManager(repo_root)
    if args.agent and args.agent.strip():
        try:
            active_profile = profile_manager.set_active_profile(args.agent.strip())
        except Exception as error:  # noqa: BLE001
            print(f"error> {error}")
            return 2
    else:
        active_profile = profile_manager.get_active_profile()

    config = RuntimeConfig(
        ui=active_profile.runtime_ui,
        provider=active_profile.runtime_provider,
        model=active_profile.runtime_model,
    )

    if config.ui not in ui_names:
        print(f"error> unknown ui '{config.ui}'")
        return 2
    if config.provider not in provider_names:
        print(f"error> unknown provider '{config.provider}'")
        return 2

    print("Active runtime:")
    print(f"- ui: {config.ui}")
    print(f"- provider: {config.provider}")
    print(f"- model: {config.model}")
    print("- trigger: interactive")
    print(f"- agent: {active_profile.name}")

    return ui_registry.run(
        config.ui,
        provider_name=config.provider,
        model=config.model,
        user_request=args.request,
        repo_root=repo_root,
        ui_options={"trigger": "interactive"},
    )


if __name__ == "__main__":
    raise SystemExit(main())
