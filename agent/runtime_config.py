"""Runtime configuration persistence and bootstrap selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent.models import ComponentCatalog
from agent.provider_registry import ProviderRegistry
from agent.ui_display import to_ui_label, to_ui_model_label


@dataclass(frozen=True)
class RuntimeConfig:
    ui: str
    provider: str
    model: str
    generation_process_log_enabled: bool = True


class RuntimeConfigStore:
    """Persistence store at `.aidzero/runtime_config.json`."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.config_file = self.repo_root / ".aidzero" / "runtime_config.json"

    def load(self) -> RuntimeConfig | None:
        if not self.config_file.exists():
            return None
        try:
            payload = json.loads(self.config_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        ui = _as_str(payload.get("ui"))
        provider = _as_str(payload.get("provider"))
        model = _as_str(payload.get("model"))
        if not ui or not provider:
            return None
        return RuntimeConfig(
            ui=ui,
            provider=provider,
            model=model,
            generation_process_log_enabled=bool(payload.get("generation_process_log_enabled", True)),
        )

    def save(self, config: RuntimeConfig) -> None:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ui": config.ui,
            "provider": config.provider,
            "model": config.model,
            "generation_process_log_enabled": config.generation_process_log_enabled,
        }
        self.config_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def ensure_runtime_config(
    *,
    catalog: ComponentCatalog,
    provider_registry: ProviderRegistry,
    store: RuntimeConfigStore,
    reconfigure: bool = False,
) -> RuntimeConfig:
    existing = None if reconfigure else store.load()
    if existing and _is_valid(existing, catalog, provider_registry):
        return existing

    ui_names = [item.name for item in catalog.ui]
    provider_names = provider_registry.names()
    if not ui_names:
        raise RuntimeError("No UI modules available.")
    if not provider_names:
        raise RuntimeError("No providers available.")

    selected_ui = _prompt_select(
        title="Select UI",
        options=ui_names,
        label_fn=to_ui_label,
    )
    selected_provider = _prompt_select(
        title="Select provider",
        options=provider_names,
        label_fn=to_ui_label,
    )
    selected_model = _prompt_select_model(provider_registry, selected_provider)
    generation_log_enabled = _prompt_yes_no(
        "Enable generation process log? [Y/n]: ",
        default=True,
    )
    config = RuntimeConfig(
        ui=selected_ui,
        provider=selected_provider,
        model=selected_model,
        generation_process_log_enabled=generation_log_enabled,
    )
    store.save(config)
    return config


def _is_valid(
    config: RuntimeConfig,
    catalog: ComponentCatalog,
    provider_registry: ProviderRegistry,
) -> bool:
    ui_names = {item.name for item in catalog.ui}
    return config.ui in ui_names and provider_registry.has(config.provider)


def _as_str(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _prompt_select(
    *,
    title: str,
    options: list[str],
    label_fn,
) -> str:
    print(f"\n{title}:")
    for index, option in enumerate(options, start=1):
        print(f"{index}. {label_fn(option)}")

    while True:
        raw = input(f"Choose 1-{len(options)} (default 1): ").strip()
        if not raw:
            return options[0]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Invalid selection. Please try again.")


def _prompt_select_model(provider_registry: ProviderRegistry, provider_name: str) -> str:
    default_model = provider_registry.default_model(provider_name)
    try:
        models = provider_registry.try_list_models(provider_name)
    except Exception:
        models = []
    unique_models: list[str] = []
    for model in models:
        if model not in unique_models:
            unique_models.append(model)
    if default_model not in unique_models:
        unique_models.insert(0, default_model)
    return _prompt_select(
        title=f"Select model for {to_ui_label(provider_name)}",
        options=unique_models,
        label_fn=to_ui_model_label,
    )


def _prompt_yes_no(prompt: str, *, default: bool) -> bool:
    default_hint = "y" if default else "n"
    while True:
        raw = input(prompt).strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print(f"Invalid answer. Type y or n (default {default_hint}).")
