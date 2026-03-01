from __future__ import annotations

from agent.engine import AgentEngine
from agent.memory import MemoryStore
from agent.models import TriggerEvent
from agent.storage import JsonlStore
from agent.tooling import ToolRegistry


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, **kwargs):
        del messages, kwargs
        self.calls += 1
        if self.calls == 1:
            return "Need a tool. <AID_TOOL_CALL>{\"name\":\"echo\",\"arguments\":{\"value\":\"ok\"}}</AID_TOOL_CALL>"
        return "Final answer from model"


def test_engine_executes_tool_and_persists_outputs(tmp_path):
    history_store = JsonlStore(tmp_path / ".aidzero" / "store" / "history.jsonl")
    output_store = JsonlStore(tmp_path / ".aidzero" / "store" / "output.jsonl")
    memory = MemoryStore(tmp_path / ".aidzero" / "memory.json")

    tools = ToolRegistry()
    tools.register(
        name="echo",
        description="Echo tool",
        parameters={"type": "object"},
        execute=lambda args: {"echo": args.get("value")},
    )

    engine = AgentEngine(
        repo_root=tmp_path,
        llm=FakeLLM(),
        tools=tools,
        history_store=history_store,
        memory_store=memory,
        output_store=output_store,
    )

    result = engine.run_event(
        TriggerEvent(kind="interactive", source="test", prompt="do something")
    )

    assert result.response == "Need a tool.\nFinal answer from model"
    assert result.used_tools == ["echo"]

    history_rows = history_store.read_all()
    output_rows = output_store.read_all()
    assert len(history_rows) == 1
    assert len(output_rows) == 1
    assert (tmp_path / ".aidzero" / "output" / "latest.txt").exists()
