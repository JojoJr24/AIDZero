"""Agent profile loading from Agents/*.json plus active profile persistence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentProfile:
    """Resolved runtime profile used to configure prompt/tools/dash modules."""

    name: str
    description: str
    system_prompt: str
    enabled_tools: list[str] | None
    enabled_dash_modules: list[str] | None
    memory_enabled: bool
    history_enabled: bool
    runtime_ui: str
    runtime_provider: str
    runtime_model: str
    source_path: Path


class AgentProfileManager:
    """Loads profile files and tracks currently active profile."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.profiles_root = self.repo_root / "Agents"
        self.state_path = self.repo_root / ".aidzero" / "agent_profile.json"

    def list_profile_names(self) -> list[str]:
        names = [path.stem for path in self._iter_profile_files()]
        return sorted(names)

    def get_active_name(self) -> str:
        names = self.list_profile_names()
        if not names:
            raise RuntimeError("No profiles found in Agents/*.json")

        persisted = self._load_state_name()
        if persisted and persisted in names:
            return persisted

        default_name = "default"
        return default_name if default_name in names else names[0]

    def get_active_profile(self) -> AgentProfile:
        return self.get_profile(self.get_active_name())

    def set_active_profile(self, name: str) -> AgentProfile:
        profile_name = name.strip()
        if not profile_name:
            raise ValueError("Profile name cannot be empty.")

        profile = self.get_profile(profile_name)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"active": profile.name}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return profile

    def get_profile(self, name: str) -> AgentProfile:
        profile_name = name.strip()
        if not profile_name:
            raise ValueError("Profile name cannot be empty.")

        path = self.profiles_root / f"{profile_name}.json"
        if not path.is_file():
            available = ", ".join(self.list_profile_names()) or "(none)"
            raise FileNotFoundError(f"Agent profile '{profile_name}' not found. Available: {available}")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid JSON in profile {path.name}: {error}") from error

        if not isinstance(payload, dict):
            raise RuntimeError(f"Profile {path.name} must contain a JSON object.")

        description = str(payload.get("description", "")).strip()

        system_prompt = self._resolve_system_prompt(payload, source_path=path)
        modules = payload.get("modules", {})
        if modules is None:
            modules = {}
        if not isinstance(modules, dict):
            raise RuntimeError(f"Invalid 'modules' in profile {path.name}; expected object.")

        enabled_tools = self._parse_module_list(modules.get("tools"), key="tools", filename=path.name)
        enabled_dash_modules = self._parse_module_list(
            modules.get("dash"),
            key="dash",
            filename=path.name,
        )
        features = payload.get("features", {})
        if features is None:
            features = {}
        if not isinstance(features, dict):
            raise RuntimeError(f"Invalid 'features' in profile {path.name}; expected object.")
        memory_enabled = self._parse_feature_flag(
            features.get("memory"),
            key="memory",
            filename=path.name,
        )
        history_enabled = self._parse_feature_flag(
            features.get("history"),
            key="history",
            filename=path.name,
        )
        runtime_ui, runtime_provider, runtime_model = self._parse_runtime_config(
            payload.get("runtime"),
            filename=path.name,
        )

        return AgentProfile(
            name=profile_name,
            description=description,
            system_prompt=system_prompt,
            enabled_tools=enabled_tools,
            enabled_dash_modules=enabled_dash_modules,
            memory_enabled=memory_enabled,
            history_enabled=history_enabled,
            runtime_ui=runtime_ui,
            runtime_provider=runtime_provider,
            runtime_model=runtime_model,
            source_path=path,
        )

    def _resolve_system_prompt(self, payload: dict[str, Any], *, source_path: Path) -> str:
        inline = payload.get("system_prompt")
        if isinstance(inline, str) and inline.strip():
            return inline.strip()

        rel = payload.get("system_prompt_file")
        if not isinstance(rel, str) or not rel.strip():
            rel = "system_prompt.md"

        prompt_path = self._resolve_agents_path(rel.strip(), source_path=source_path)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
        text = prompt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            raise RuntimeError(f"System prompt file is empty: {prompt_path}")
        return text

    def _parse_module_list(self, raw: Any, *, key: str, filename: str) -> list[str] | None:
        if raw is None:
            return None
        if isinstance(raw, str):
            value = raw.strip().lower()
            if value == "all":
                return None
            raise RuntimeError(f"Invalid modules.{key} in {filename}: expected 'all' or string list.")

        if not isinstance(raw, list):
            raise RuntimeError(f"Invalid modules.{key} in {filename}: expected list or 'all'.")

        names: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise RuntimeError(f"Invalid modules.{key} item in {filename}: expected non-empty string.")
            candidate = item.strip()
            if candidate not in names:
                names.append(candidate)
        return names

    def _parse_feature_flag(self, raw: Any, *, key: str, filename: str) -> bool:
        if raw is None:
            return True
        if isinstance(raw, bool):
            return raw
        raise RuntimeError(f"Invalid features.{key} in {filename}: expected boolean.")

    def _parse_runtime_config(self, raw: Any, *, filename: str) -> tuple[str, str, str]:
        if not isinstance(raw, dict):
            raise RuntimeError(f"Invalid runtime in {filename}: expected object with ui/provider/model.")

        ui = str(raw.get("ui", "")).strip()
        provider = str(raw.get("provider", "")).strip()
        model = str(raw.get("model", "")).strip()
        if not ui or not provider or not model:
            raise RuntimeError(f"Invalid runtime in {filename}: ui/provider/model are required non-empty strings.")
        return ui, provider, model

    def _load_state_name(self) -> str | None:
        if not self.state_path.exists():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        raw = payload.get("active")
        if not isinstance(raw, str):
            return None
        name = raw.strip()
        return name or None

    def _iter_profile_files(self) -> list[Path]:
        if not self.profiles_root.is_dir():
            return []
        return sorted(
            [path for path in self.profiles_root.glob("*.json") if path.is_file()],
            key=lambda item: item.name.lower(),
        )

    def _resolve_agents_path(self, raw_path: str, *, source_path: Path) -> Path:
        base = source_path.parent
        candidate = (base / raw_path).resolve()
        if self._is_within_agents(candidate):
            return candidate
        raise RuntimeError(f"System prompt path must stay within Agents/: {raw_path}")

    def _is_within_agents(self, path: Path) -> bool:
        try:
            path.relative_to(self.profiles_root)
            return True
        except ValueError:
            return False
