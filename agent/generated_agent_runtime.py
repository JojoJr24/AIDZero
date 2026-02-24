"""Runtime helpers copied into generated agents."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, cast

from .provider_base import LLMProvider

DEFAULT_AGENT_CONFIG_FILENAME = "agent_config.json"
REQUIRED_PROVIDER_METHODS = ["generate_text", "list_model_names"]


def load_runtime_config(
    *,
    project_root: Path,
    config_filename: str = DEFAULT_AGENT_CONFIG_FILENAME,
) -> dict[str, Any]:
    """Load generated agent runtime config from project root."""
    config_path = project_root / config_filename
    if not config_path.exists():
        raise FileNotFoundError(f"Runtime config file not found: {config_path}")

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Runtime config is not valid JSON: {config_path}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"Runtime config must be a JSON object: {config_path}")
    return payload


def create_provider_from_config(*, project_root: Path, config: Mapping[str, Any]) -> LLMProvider:
    """Create provider instance from runtime config payload."""
    provider_name = _required_string(config.get("provider"), label="provider")
    provider_options = config.get("provider_options", {})
    if provider_options is None:
        provider_options = {}
    if not isinstance(provider_options, dict):
        raise ValueError("Runtime config field 'provider_options' must be a JSON object.")

    module_path = project_root / "LLMProviders" / provider_name / "provider.py"
    module_name = f"generated_agent_provider_{_sanitize_module_suffix(provider_name)}"
    module = _load_python_module(module_name=module_name, module_path=module_path)
    provider_class = _resolve_provider_class(module=module, module_path=module_path)

    try:
        instance = provider_class(**provider_options)
    except TypeError as error:
        raise RuntimeError(
            f"Failed to instantiate provider class '{provider_class.__name__}' "
            f"from {module_path}: {error}"
        ) from error
    return cast(LLMProvider, instance)


def _required_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Runtime config field '{label}' is required and must be a non-empty string.")
    return value.strip()


def _sanitize_module_suffix(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower() or "provider"


def _load_python_module(*, module_name: str, module_path: Path) -> ModuleType:
    if not module_path.exists():
        raise FileNotFoundError(f"Provider module not found: {module_path}")

    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not create module spec for provider file: {module_path}")

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _resolve_provider_class(*, module: ModuleType, module_path: Path) -> type:
    explicit_name = getattr(module, "PROVIDER_CLASS", None)
    if isinstance(explicit_name, str) and explicit_name.strip():
        explicit_class = getattr(module, explicit_name.strip(), None)
        if inspect.isclass(explicit_class):
            return cast(type, explicit_class)
        raise RuntimeError(
            f"Provider module {module_path} defines PROVIDER_CLASS='{explicit_name}', "
            "but that class was not found."
        )

    local_classes = [
        cast(type, candidate)
        for candidate in vars(module).values()
        if inspect.isclass(candidate) and candidate.__module__ == module.__name__
    ]
    provider_candidates = [candidate for candidate in local_classes if _is_provider_candidate(candidate)]
    if not provider_candidates:
        class_names = [candidate.__name__ for candidate in local_classes]
        raise RuntimeError(
            f"Provider module {module_path} does not expose a usable provider class. "
            f"Found classes: {class_names or 'none'}."
        )

    scored: list[tuple[int, type]] = [
        (_provider_method_score(candidate), candidate) for candidate in provider_candidates
    ]
    scored.sort(key=lambda item: (item[0], item[1].__name__), reverse=True)
    best_score = scored[0][0]
    if best_score <= 0:
        class_names = [candidate.__name__ for candidate in provider_candidates]
        raise RuntimeError(
            f"Provider module {module_path} contains candidate classes {class_names}, "
            "but none implement the expected provider interface."
        )

    best_candidates = [candidate for score, candidate in scored if score == best_score]
    if len(best_candidates) == 1:
        return best_candidates[0]

    for candidate in best_candidates:
        if candidate.__name__ == "Provider":
            return candidate
    best_names = [candidate.__name__ for candidate in best_candidates]
    raise RuntimeError(
        f"Provider module {module_path} has multiple matching provider classes: {best_names}. "
        "Set PROVIDER_CLASS in that module to disambiguate."
    )


def _is_provider_candidate(candidate: type) -> bool:
    if candidate.__name__.endswith("Error"):
        return False
    if issubclass(candidate, BaseException):
        return False
    return candidate.__name__.endswith("Provider")


def _provider_method_score(candidate: type) -> int:
    score = 0
    for method_name in REQUIRED_PROVIDER_METHODS:
        method = getattr(candidate, method_name, None)
        if callable(method):
            score += 1
    return score
