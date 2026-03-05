"""Shared runtime builder used by UI modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from CORE.agents import AgentProfile, AgentProfileManager
from CORE.engine import AgentEngine
from CORE.gateway import TriggerGateway
from CORE.llm_client import LLMClient
from CORE.memory import MemoryStore
from CORE.prompt_history import PromptHistoryStore
from CORE.storage import JsonlStore
from CORE.tooling import build_default_tool_registry

MEMORY_TOOL_NAMES = {"memory_get", "memory_set", "memory_list"}
HISTORY_TOOL_NAMES = {"history_get"}


@dataclass(frozen=True)
class UIRuntime:
    engine: AgentEngine
    gateway: TriggerGateway
    history: PromptHistoryStore
    agent_manager: AgentProfileManager
    agent_profile: AgentProfile


def build_ui_runtime(*, repo_root: Path, provider_name: str, model: str) -> UIRuntime:
    root = repo_root.resolve()

    llm = LLMClient(repo_root=root, provider_name=provider_name, model=model)
    memory = MemoryStore(root / ".aidzero" / "memory.json")
    agent_manager = AgentProfileManager(root)
    agent_profile = agent_manager.get_active_profile()
    memory.enabled = agent_profile.memory_enabled
    disabled_tools = profile_disabled_tools(agent_profile)
    tools = build_default_tool_registry(
        root,
        memory,
        enabled_names=agent_profile.enabled_tools,
        disabled_names=disabled_tools,
    )
    history_store = JsonlStore(root / ".aidzero" / "store" / "history.jsonl")
    output_store = JsonlStore(root / ".aidzero" / "store" / "output.jsonl")

    engine = AgentEngine(
        repo_root=root,
        llm=llm,
        tools=tools,
        history_store=history_store,
        memory_store=memory,
        output_store=output_store,
        system_prompt_override=agent_profile.system_prompt,
        history_enabled=agent_profile.history_enabled,
    )

    return UIRuntime(
        engine=engine,
        gateway=TriggerGateway(root),
        history=PromptHistoryStore(root, enabled=agent_profile.history_enabled),
        agent_manager=agent_manager,
        agent_profile=agent_profile,
    )


def profile_disabled_tools(agent_profile: AgentProfile) -> list[str]:
    disabled: set[str] = set()
    if not agent_profile.memory_enabled:
        disabled.update(MEMORY_TOOL_NAMES)
    if not agent_profile.history_enabled:
        disabled.update(HISTORY_TOOL_NAMES)
    return sorted(disabled)
