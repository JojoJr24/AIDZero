"""Discovery and factory utilities for provider adapters."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

from LLMProviders.provider_base import LLMProvider

DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    "AID-openai": "gpt-4o-mini",
    "AID-google_gemini": "gemini-2.5-flash",
    "AID-claude": "claude-3-5-sonnet-latest",
    "AID-ollama": "llama3.2",
    "AID-lmstudio": "gpt-4o-mini",
    "AID-llamacpp": "llama3.2",
}


class ProviderRegistry:
    """Load and instantiate provider adapters from `LLMProviders/AID-*/provider.py`."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.providers_root = self.repo_root / "LLMProviders"

    def names(self) -> list[str]:
        names: list[str] = []
        if not self.providers_root.is_dir():
            return names
        for entry in sorted(self.providers_root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir() or not entry.name.startswith("AID-"):
                continue
            if (entry / "provider.py").is_file():
                names.append(entry.name)
        return names

    def has(self, provider_name: str) -> bool:
        return provider_name.strip() in set(self.names())

    def default_model(self, provider_name: str) -> str:
        normalized = provider_name.strip()
        if normalized in DEFAULT_MODEL_BY_PROVIDER:
            return DEFAULT_MODEL_BY_PROVIDER[normalized]
        return "gpt-4o-mini"

    def try_list_models(self, provider_name: str) -> list[str]:
        provider = self.create(provider_name)
        return provider.list_model_names()

    def create(self, provider_name: str) -> LLMProvider:
        normalized_name = provider_name.strip()
        if not normalized_name:
            raise ValueError("provider_name cannot be empty.")
        module_path = self.providers_root / normalized_name / "provider.py"
        if not module_path.is_file():
            raise FileNotFoundError(f"Provider module not found: {module_path}")

        module_name = f"aidzero_provider_{normalized_name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load provider module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        provider_cls = _find_provider_class(module)
        provider = provider_cls()
        return provider


def _find_provider_class(module: Any) -> type[LLMProvider]:
    candidates: list[type[Any]] = []
    for _, member in inspect.getmembers(module, inspect.isclass):
        if member.__module__ != module.__name__:
            continue
        if not member.__name__.endswith("Provider"):
            continue
        if member.__name__ == "OpenAICompatibleProvider":
            continue
        candidates.append(member)
    if not candidates:
        raise RuntimeError(f"No provider class found in module '{module.__name__}'.")
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda item: item.__name__)
    return candidates[0]
