"""Provider registration and factory helpers for the AIDZero runtime."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProviderSpec:
    """Single provider registration entry."""

    name: str
    entrypoint: Path
    class_name: str
    default_model: str


class ProviderRegistry:
    """Creates provider instances from provider names."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self._providers: dict[str, ProviderSpec] = {
            "AID-google_gemini": ProviderSpec(
                name="AID-google_gemini",
                entrypoint=self.repo_root / "LLMProviders" / "AID-google_gemini" / "provider.py",
                class_name="GeminiProvider",
                default_model="gemini-2.5-flash",
            ),
            "AID-openai": ProviderSpec(
                name="AID-openai",
                entrypoint=self.repo_root / "LLMProviders" / "AID-openai" / "provider.py",
                class_name="OpenAIProvider",
                default_model="gpt-4o-mini",
            ),
            "AID-claude": ProviderSpec(
                name="AID-claude",
                entrypoint=self.repo_root / "LLMProviders" / "AID-claude" / "provider.py",
                class_name="ClaudeProvider",
                default_model="claude-3-5-sonnet-latest",
            ),
            "AID-ollama": ProviderSpec(
                name="AID-ollama",
                entrypoint=self.repo_root / "LLMProviders" / "AID-ollama" / "provider.py",
                class_name="OllamaProvider",
                default_model="llama3.1",
            ),
            "AID-lmstudio": ProviderSpec(
                name="AID-lmstudio",
                entrypoint=self.repo_root / "LLMProviders" / "AID-lmstudio" / "provider.py",
                class_name="LMStudioProvider",
                default_model="local-model",
            ),
            "AID-llamacpp": ProviderSpec(
                name="AID-llamacpp",
                entrypoint=self.repo_root / "LLMProviders" / "AID-llamacpp" / "provider.py",
                class_name="LlamaCppProvider",
                default_model="local-model",
            ),
        }

    def names(self) -> list[str]:
        return sorted(self._providers.keys())

    def has(self, provider_name: str) -> bool:
        return provider_name in self._providers

    def create(self, provider_name: str) -> Any:
        spec = self._providers.get(provider_name)
        if spec is None:
            raise ValueError(f"Unknown provider: {provider_name}")
        provider_class = self._load_provider_class(spec)
        return provider_class()

    def default_model(self, provider_name: str) -> str:
        spec = self._providers.get(provider_name)
        if spec is None:
            raise ValueError(f"Unknown provider: {provider_name}")
        return spec.default_model

    def try_list_models(self, provider_name: str) -> list[str]:
        provider = self.create(provider_name)
        model_names = provider.list_model_names()
        return [name for name in model_names if isinstance(name, str) and name.strip()]

    @staticmethod
    def _load_provider_class(spec: ProviderSpec) -> type:
        if not spec.entrypoint.exists():
            raise FileNotFoundError(f"Provider entrypoint not found: {spec.entrypoint}")

        module_name = _module_name_from_provider(spec.name)
        module_spec = importlib.util.spec_from_file_location(module_name, spec.entrypoint)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Could not load provider module from: {spec.entrypoint}")

        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        provider_class = getattr(module, spec.class_name, None)
        if provider_class is None:
            raise AttributeError(
                f"Provider class '{spec.class_name}' not found in {spec.entrypoint}"
            )
        return provider_class


def _module_name_from_provider(provider_name: str) -> str:
    sanitized = provider_name.lower().replace("-", "_")
    return f"aidzero_provider_{sanitized}"
