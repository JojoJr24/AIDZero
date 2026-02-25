"""Runtime configuration persisted between agent executions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .models import ComponentCatalog
from .provider_registry import ProviderRegistry
from .ui_display import to_ui_label, to_ui_model_label

CONFIG_DIR = ".aidzero"
LEGACY_CONFIG_DIR = ".autoagent"
CONFIG_FILENAME = "runtime_config.json"


@dataclass
class RuntimeConfig:
    """Startup runtime configuration selected by the user."""

    ui: str
    provider: str
    model: str
    generation_process_log_enabled: bool = True


class RuntimeConfigStore:
    """Load/save runtime config under repository root."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.path = self.repo_root / CONFIG_DIR / CONFIG_FILENAME
        self.legacy_path = self.repo_root / LEGACY_CONFIG_DIR / CONFIG_FILENAME

    def load(self) -> RuntimeConfig | None:
        for candidate in (self.path, self.legacy_path):
            config = self._load_from_path(candidate)
            if config:
                return config
        return None

    def save(self, config: RuntimeConfig) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2, ensure_ascii=False)
        return self.path

    @staticmethod
    def _load_from_path(path: Path) -> RuntimeConfig | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None
        generation_process_log_enabled = _coerce_bool(
            payload.get("generation_process_log_enabled"), default=True
        )
        return RuntimeConfig(
            ui=str(payload.get("ui", "")).strip(),
            provider=str(payload.get("provider", "")).strip(),
            model=str(payload.get("model", "")).strip(),
            generation_process_log_enabled=generation_process_log_enabled,
        )


def ensure_runtime_config(
    *,
    catalog: ComponentCatalog,
    provider_registry: ProviderRegistry,
    store: RuntimeConfigStore,
    reconfigure: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> RuntimeConfig:
    """Load existing config or run the first-time interactive setup."""
    if not reconfigure:
        existing = store.load()
        if existing and _is_valid(existing, catalog, provider_registry):
            return existing

    config = _run_first_time_setup(
        catalog=catalog,
        provider_registry=provider_registry,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    saved_path = store.save(config)
    output_fn(f"Saved runtime configuration to: {saved_path}")
    return config


def _run_first_time_setup(
    *,
    catalog: ComponentCatalog,
    provider_registry: ProviderRegistry,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> RuntimeConfig:
    output_fn("First-time setup: choose UI, provider, and model.")

    available_ui = [item.name for item in catalog.ui]
    if not available_ui:
        available_ui = ["terminal"]
    selected_ui = _select_option(
        label="UI",
        options=available_ui,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    catalog_provider_names = {item.name for item in catalog.llm_providers}
    selectable_providers = [name for name in provider_registry.names() if name in catalog_provider_names]
    if not selectable_providers:
        raise RuntimeError("No compatible providers found in LLMProviders.")
    selected_provider = _select_option(
        label="provider",
        options=selectable_providers,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    model_names: list[str]
    try:
        model_names = provider_registry.try_list_models(selected_provider)
    except Exception as error:  # noqa: BLE001
        output_fn(f"Could not list models automatically ({error}).")
        model_names = []

    if model_names:
        selected_model = _select_option(
            label="model",
            options=model_names,
            input_fn=input_fn,
            output_fn=output_fn,
            display_fn=to_ui_model_label,
        )
    else:
        default_model = provider_registry.default_model(selected_provider)
        raw_model = input_fn(f"Enter model name [{default_model}]: ").strip()
        selected_model = raw_model or default_model

    output_fn(
        "Selected configuration:\n"
        f"- ui: {to_ui_label(selected_ui)}\n"
        f"- provider: {to_ui_label(selected_provider)}\n"
        f"- model: {to_ui_model_label(selected_model)}"
    )
    confirm = input_fn("Save this configuration? [Y/n]: ").strip().lower()
    if confirm in {"n", "no"}:
        return _run_first_time_setup(
            catalog=catalog,
            provider_registry=provider_registry,
            input_fn=input_fn,
            output_fn=output_fn,
        )

    return RuntimeConfig(
        ui=selected_ui,
        provider=selected_provider,
        model=selected_model,
        generation_process_log_enabled=True,
    )


def _select_option(
    *,
    label: str,
    options: list[str],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    display_fn: Callable[[str], str] = to_ui_label,
) -> str:
    output_fn(f"Available {label} options:")
    for index, option in enumerate(options, start=1):
        output_fn(f"  {index}. {display_fn(option)}")
    while True:
        raw = input_fn(f"Select {label} [1-{len(options)}]: ").strip()
        if not raw:
            return options[0]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        output_fn(f"Invalid selection for {label}.")


def _is_valid(
    config: RuntimeConfig,
    catalog: ComponentCatalog,
    provider_registry: ProviderRegistry,
) -> bool:
    if not config.ui or not config.provider or not config.model:
        return False
    ui_names = {item.name for item in catalog.ui}
    if ui_names and config.ui not in ui_names:
        return False
    if not provider_registry.has(config.provider):
        return False
    provider_names = {item.name for item in catalog.llm_providers}
    if provider_names and config.provider not in provider_names:
        return False
    return True


def _coerce_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default
