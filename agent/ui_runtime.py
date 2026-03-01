"""Shared runtime builder used by UI modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.engine import AgentEngine
from agent.gateway import TriggerGateway
from agent.llm_client import LLMClient
from agent.memory import MemoryStore
from agent.prompt_history import PromptHistoryStore
from agent.storage import JsonlStore
from agent.tooling import build_default_tool_registry


@dataclass(frozen=True)
class UIRuntime:
    engine: AgentEngine
    gateway: TriggerGateway
    history: PromptHistoryStore


def build_ui_runtime(*, repo_root: Path, provider_name: str, model: str) -> UIRuntime:
    root = repo_root.resolve()

    llm = LLMClient(repo_root=root, provider_name=provider_name, model=model)
    memory = MemoryStore(root / ".aidzero" / "memory.json")
    tools = build_default_tool_registry(root, memory)
    history_store = JsonlStore(root / ".aidzero" / "store" / "history.jsonl")
    output_store = JsonlStore(root / ".aidzero" / "store" / "output.jsonl")

    engine = AgentEngine(
        repo_root=root,
        llm=llm,
        tools=tools,
        history_store=history_store,
        memory_store=memory,
        output_store=output_store,
    )

    return UIRuntime(
        engine=engine,
        gateway=TriggerGateway(root),
        history=PromptHistoryStore(root),
    )
