"""Shared serialization helpers for core API transport."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from CORE.agents import AgentProfile, resolve_agents_root
from CORE.models import TriggerEvent, TurnResult, utc_now_iso


def trigger_event_to_dict(event: TriggerEvent) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "source": event.source,
        "prompt": event.prompt,
        "created_at": event.created_at,
        "metadata": dict(event.metadata),
    }


def trigger_event_from_dict(payload: dict[str, Any]) -> TriggerEvent:
    created_at = str(payload.get("created_at") or "").strip()
    return TriggerEvent(
        kind=str(payload.get("kind", "interactive")),
        source=str(payload.get("source", "api")),
        prompt=str(payload.get("prompt", "")),
        created_at=created_at or utc_now_iso(),
        metadata=_as_dict(payload.get("metadata")),
    )


def turn_result_to_dict(result: TurnResult) -> dict[str, Any]:
    return {
        "event": trigger_event_to_dict(result.event),
        "response": result.response,
        "rounds": result.rounds,
        "used_tools": list(result.used_tools),
    }


def turn_result_from_dict(payload: dict[str, Any]) -> TurnResult:
    event_payload = _as_dict(payload.get("event"))
    return TurnResult(
        event=trigger_event_from_dict(event_payload),
        response=str(payload.get("response", "")),
        rounds=int(payload.get("rounds", 0)),
        used_tools=[str(item) for item in payload.get("used_tools", []) if str(item).strip()],
    )


def profile_to_dict(profile: AgentProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "description": profile.description,
        "system_prompt": profile.system_prompt,
        "enabled_tools": profile.enabled_tools,
        "enabled_dash_modules": profile.enabled_dash_modules,
        "memory_enabled": profile.memory_enabled,
        "history_enabled": profile.history_enabled,
        "runtime_ui": profile.runtime_ui,
        "runtime_provider": profile.runtime_provider,
        "runtime_model": profile.runtime_model,
        "source_path": str(profile.source_path),
    }


def profile_from_dict(payload: dict[str, Any], *, repo_root: Path) -> AgentProfile:
    source_raw = str(payload.get("source_path", "")).strip() or "Agents/default/default.json"
    source_path = Path(source_raw)
    if not source_path.is_absolute():
        if source_path.parts and source_path.parts[0] == "Agents":
            source_path = (resolve_agents_root(repo_root) / Path(*source_path.parts[1:])).resolve()
        else:
            source_path = (repo_root / source_path).resolve()
    return AgentProfile(
        name=str(payload.get("name", "default")),
        description=str(payload.get("description", "")),
        system_prompt=str(payload.get("system_prompt", "")),
        enabled_tools=_as_optional_list(payload.get("enabled_tools")),
        enabled_dash_modules=_as_optional_list(payload.get("enabled_dash_modules")),
        memory_enabled=bool(payload.get("memory_enabled", True)),
        history_enabled=bool(payload.get("history_enabled", True)),
        runtime_ui=str(payload.get("runtime_ui", "terminal")).strip() or "terminal",
        runtime_provider=str(payload.get("runtime_provider", "openai")).strip() or "openai",
        runtime_model=str(payload.get("runtime_model", "gpt-4o-mini")).strip() or "gpt-4o-mini",
        source_path=source_path,
    )


def _as_optional_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}
