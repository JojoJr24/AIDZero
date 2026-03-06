#!/usr/bin/env python3
"""Root runtime launcher for the AIDZero agent runtime project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent
CODE_ROOT = REPO_ROOT / "AIDZeroCode"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from CORE.agents import AgentProfileManager
from CORE.api_server import serve_core_api
from CORE.models import RuntimeConfig
from CORE.repo_layout import resolve_code_root
from CORE.ui_runtime import build_ui_runtime
from CORE.ui_registry import UIRegistry


@dataclass(frozen=True)
class CliArgs:
    request: str | None
    agent: str | None
    headless: bool


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="AIDZero runtime launcher.")
    parser.add_argument("--request", help="Prompt to process.")
    parser.add_argument("--agent", default=None, help="Agent profile name from Agents/*.json.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run one-shot using HeadlessPrompt.txt and save output in Results/.",
    )
    parsed = parser.parse_args()
    return CliArgs(
        request=parsed.request,
        agent=parsed.agent,
        headless=bool(parsed.headless),
    )


def _discover_providers(repo_root: Path) -> list[str]:
    root = resolve_code_root(repo_root) / "LLMProviders"
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
    provider_file = resolve_code_root(repo_root) / "LLMProviders" / provider_name / "provider.py"
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


def _read_headless_prompt(repo_root: Path) -> str:
    prompt_file = repo_root / "HeadlessPrompt.txt"
    if not prompt_file.is_file():
        raise FileNotFoundError(f"Headless prompt file not found: {prompt_file}")
    prompt = prompt_file.read_text(encoding="utf-8", errors="replace").strip()
    if not prompt:
        raise ValueError(f"Headless prompt file is empty: {prompt_file}")
    return prompt


def _write_headless_result(repo_root: Path, response: str) -> Path:
    results_dir = repo_root / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = results_dir / f"result_{now}.txt"
    latest_path = results_dir / "latest.txt"
    payload = response.rstrip() + "\n"

    timestamped_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return timestamped_path


def _run_headless(repo_root: Path, *, provider: str, model: str, prompt: str) -> int:
    runtime = build_ui_runtime(repo_root=repo_root, provider_name=provider, model=model)
    events = runtime.gateway.collect(trigger="interactive", prompt=prompt)
    if not events:
        print("error> no events collected for headless prompt")
        return 2

    outputs: list[dict[str, object]] = []
    for event in events:
        turn = runtime.engine.run_event(event)
        outputs.append(
            {
                "kind": event.kind,
                "source": event.source,
                "response": turn.response,
                "rounds": turn.rounds,
                "used_tools": turn.used_tools,
            }
        )

    if len(outputs) == 1:
        response_text = str(outputs[0]["response"])
    else:
        response_text = json.dumps(outputs, ensure_ascii=False, indent=2)

    output_path = _write_headless_result(repo_root, response_text)
    print("Headless runtime:")
    print(f"- agent: {runtime.agent_profile.name}")
    print(f"- provider: {provider}")
    print(f"- model: {model}")
    print(f"- results: {output_path}")
    return 0


def main() -> int:
    args = _parse_args()
    repo_root = REPO_ROOT

    ui_registry = UIRegistry(repo_root)
    ui_names = ui_registry.names()
    provider_names = _discover_providers(repo_root)

    profile_manager = AgentProfileManager(repo_root)
    if args.headless:
        try:
            active_profile = profile_manager.set_active_profile("default")
        except Exception as error:  # noqa: BLE001
            print(f"error> could not load default agent profile: {error}")
            return 2
        if args.agent and args.agent.strip():
            print("warning> --agent is ignored in --headless mode (using 'default').")
    else:
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

    if args.headless:
        try:
            prompt = _read_headless_prompt(repo_root)
        except Exception as error:  # noqa: BLE001
            print(f"error> {error}")
            return 2
        try:
            return _run_headless(
                repo_root,
                provider=config.provider,
                model=config.model,
                prompt=prompt,
            )
        except Exception as error:  # noqa: BLE001
            print(f"error> headless execution failed: {error}")
            return 2

    try:
        ui_type = ui_registry.ui_type(config.ui)
    except FileNotFoundError:
        print(f"error> unknown ui '{config.ui}'")
        return 2

    if ui_type == "thirdparty":
        bind_host = "0.0.0.0"
        bind_port = 8765
        if args.request:
            print("warning> --request is ignored for thirdparty UI mode.")
        print("Third-party UI runtime:")
        print(f"- ui: {config.ui}")
        print(f"- core_api: http://{bind_host}:{bind_port}")
        print(f"- provider: {config.provider}")
        print(f"- model: {config.model}")
        print(f"- agent: {active_profile.name}")
        try:
            return serve_core_api(
                repo_root=repo_root,
                provider_name=config.provider,
                model=config.model,
                host=bind_host,
                port=bind_port,
            )
        except RuntimeError as error:
            print(f"error> {error}")
            print("tip> Cerrá el proceso que usa ese puerto o ejecutá aidzero-core con --port 8766.")
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
