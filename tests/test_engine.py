from __future__ import annotations

import json

from core.engine import AgentEngine
from core.memory import MemoryStore
from core.models import TriggerEvent
from core.storage import JsonlStore
from core.tooling import ToolRegistry


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.messages_by_call = []

    def complete(self, messages, **kwargs):
        del kwargs
        self.calls += 1
        self.messages_by_call.append(messages)
        if self.calls == 1:
            return "Need a tool. <AID_TOOL_CALL>{\"name\":\"echo\",\"arguments\":{\"value\":\"ok\"}}</AID_TOOL_CALL>"
        return "Final answer from model"


class StreamingFakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, **kwargs):
        del messages, kwargs
        raise AssertionError("complete should not be used in this test")

    def complete_stream(self, messages, **kwargs):
        del messages, kwargs
        self.calls += 1
        if self.calls == 1:
            yield "Preface <think>hidden reasoning</think>"
            yield "<AID_TOO"
            yield "L_CALL>{\"name\":\"echo\",\"arguments\":{\"value\":\"ok\"}}</AID_TOOL_CALL>"
            return
        yield "Final text"


class LoopingToolCallLLM:
    def complete(self, messages, **kwargs):
        del messages, kwargs
        return "<AID_TOOL_CALL>{\"name\":\"echo\",\"arguments\":{\"value\":\"ok\"}}</AID_TOOL_CALL>"


class SnapshotStreamingLLM:
    def complete(self, messages, **kwargs):
        del messages, kwargs
        raise AssertionError("complete should not be used in this test")

    def complete_stream(self, messages, **kwargs):
        del messages, kwargs
        yield "H"
        yield "Ho"
        yield "Hol"
        yield "Hola"


class SequentialLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.messages_by_call: list[list[dict]] = []
        self.calls = 0

    def complete(self, messages, **kwargs):
        del kwargs
        self.messages_by_call.append(messages)
        response = self.responses[self.calls]
        self.calls += 1
        return response


class InfiniteSameChunkStreamingLLM:
    def __init__(self) -> None:
        self.stopped = False

    def complete(self, messages, **kwargs):
        del messages, kwargs
        raise AssertionError("complete should not be used in this test")

    def complete_stream(self, messages, **kwargs):
        del messages, kwargs
        while not self.stopped:
            yield "Hola"

    def stop_stream(self) -> None:
        self.stopped = True


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

    first_call_messages = engine.llm.messages_by_call[0]
    system_content = first_call_messages[0]["content"]
    assert "Runtime context:" in system_content
    assert "- OS:" in system_content
    assert "- Local time:" in system_content
    assert "- UTC time:" in system_content
    payload_text = system_content.split(
        "Internal runtime payload (not user prompt; do not discuss it unless asked):\n",
        1,
    )[1]
    payload = json.loads(payload_text)
    assert "jsonl_history" not in payload
    first_user_content = first_call_messages[1]["content"]
    assert "Internal runtime payload" not in first_user_content


def test_engine_adds_linux_details_in_system_message(tmp_path, monkeypatch):
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
    monkeypatch.setattr("core.engine.platform.system", lambda: "Linux")
    monkeypatch.setattr("core.engine.platform.release", lambda: "6.8.0-test")
    monkeypatch.setattr(
        "core.engine.platform.freedesktop_os_release",
        lambda: {"NAME": "Ubuntu", "VERSION_ID": "24.04"},
    )

    engine = AgentEngine(
        repo_root=tmp_path,
        llm=FakeLLM(),
        tools=tools,
        history_store=history_store,
        memory_store=memory,
        output_store=output_store,
    )
    engine.run_event(TriggerEvent(kind="interactive", source="test", prompt="do something"))

    first_call_messages = engine.llm.messages_by_call[0]
    system_content = first_call_messages[0]["content"]
    assert "- Linux distribution: Ubuntu 24.04" in system_content
    assert "- Linux kernel: 6.8.0-test" in system_content


def test_engine_never_injects_memory_into_payload(tmp_path):
    history_store = JsonlStore(tmp_path / ".aidzero" / "store" / "history.jsonl")
    output_store = JsonlStore(tmp_path / ".aidzero" / "store" / "output.jsonl")
    memory = MemoryStore(tmp_path / ".aidzero" / "memory.json")
    memory.set("session_note", "keep me")

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

    engine.run_event(TriggerEvent(kind="interactive", source="test", prompt="hola"))
    first_call_messages = engine.llm.messages_by_call[0]
    first_payload_text = first_call_messages[0]["content"].split(
        "Internal runtime payload (not user prompt; do not discuss it unless asked):\n",
        1,
    )[1]
    first_payload = json.loads(first_payload_text)
    assert "memory" not in first_payload
    assert "memory_included" not in first_payload

    engine.run_event(
        TriggerEvent(
            kind="interactive",
            source="test",
            prompt="recuerda este contexto para responder",
        )
    )
    second_call_messages = engine.llm.messages_by_call[2]
    second_payload_text = second_call_messages[0]["content"].split(
        "Internal runtime payload (not user prompt; do not discuss it unless asked):\n",
        1,
    )[1]
    second_payload = json.loads(second_payload_text)
    assert "memory" not in second_payload
    assert "memory_included" not in second_payload


def test_engine_does_not_include_memory_for_generic_memory_words(tmp_path):
    history_store = JsonlStore(tmp_path / ".aidzero" / "store" / "history.jsonl")
    output_store = JsonlStore(tmp_path / ".aidzero" / "store" / "output.jsonl")
    memory = MemoryStore(tmp_path / ".aidzero" / "memory.json")
    memory.set("session_note", "keep me")

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
    engine.run_event(
        TriggerEvent(
            kind="interactive",
            source="test",
            prompt="explica memory leak en python",
        )
    )

    first_call_messages = engine.llm.messages_by_call[0]
    payload_text = first_call_messages[0]["content"].split(
        "Internal runtime payload (not user prompt; do not discuss it unless asked):\n",
        1,
    )[1]
    payload = json.loads(payload_text)
    assert "memory" not in payload
    assert "memory_included" not in payload


def test_engine_streams_visible_text_and_filters_tool_blocks(tmp_path):
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
        llm=StreamingFakeLLM(),
        tools=tools,
        history_store=history_store,
        memory_store=memory,
        output_store=output_store,
    )

    chunks: list[str] = []
    artifacts: list[dict] = []
    result = engine.run_event(
        TriggerEvent(kind="interactive", source="test", prompt="stream now"),
        on_stream=chunks.append,
        on_artifact=artifacts.append,
    )

    assert result.response == "Preface\nFinal text"
    streamed = "".join(chunks)
    assert streamed == "Preface Final text"
    assert "<AID_TOOL_CALL>" not in streamed
    assert "<think>" not in streamed
    starts = [item for item in artifacts if item.get("event") == "start"]
    assert [item["type"] for item in starts] == ["think", "tool_call"]
    assert [item["offset"] for item in starts] == [8, 8]
    assert any(item.get("event") == "chunk" and item.get("artifact_id") == "r1-a1" for item in artifacts)
    assert any(item.get("event") == "end" and item.get("artifact_id") == "r1-a2" for item in artifacts)


def test_engine_breaks_repeated_identical_tool_call_loop(tmp_path):
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
        llm=LoopingToolCallLLM(),
        tools=tools,
        history_store=history_store,
        memory_store=memory,
        output_store=output_store,
    )

    result = engine.run_event(
        TriggerEvent(kind="interactive", source="test", prompt="loop"),
        max_rounds=20,
    )

    assert "repeated identical tool call detected" in result.response
    assert result.rounds == 3


def test_engine_executes_legacy_tool_call_format(tmp_path):
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
    llm = SequentialLLM(
        [
            "<tool_call>echo<arg_key>value</arg_key><arg_value>ok</arg_value></tool_call>",
            "Final legacy answer",
        ]
    )
    engine = AgentEngine(
        repo_root=tmp_path,
        llm=llm,
        tools=tools,
        history_store=history_store,
        memory_store=memory,
        output_store=output_store,
    )

    result = engine.run_event(TriggerEvent(kind="interactive", source="test", prompt="legacy"))

    assert result.used_tools == ["echo"]
    assert result.response == "Final legacy answer"


def test_engine_executes_legacy_tool_call_with_wrapped_tag_name(tmp_path):
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
    llm = SequentialLLM(
        [
            "<too\nl_call>echo<arg_key>value</arg_key><arg_value>ok</arg_value></too\nl_call>",
            "Final wrapped answer",
        ]
    )
    engine = AgentEngine(
        repo_root=tmp_path,
        llm=llm,
        tools=tools,
        history_store=history_store,
        memory_store=memory,
        output_store=output_store,
    )

    result = engine.run_event(TriggerEvent(kind="interactive", source="test", prompt="legacy wrapped"))

    assert result.used_tools == ["echo"]
    assert result.response == "Final wrapped answer"


def test_engine_handles_snapshot_stream_without_duplication(tmp_path):
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
        llm=SnapshotStreamingLLM(),
        tools=tools,
        history_store=history_store,
        memory_store=memory,
        output_store=output_store,
    )

    chunks: list[str] = []
    result = engine.run_event(
        TriggerEvent(kind="interactive", source="test", prompt="saluda"),
        on_stream=chunks.append,
    )
    assert result.response == "Hola"
    assert "".join(chunks) == "Hola"


def test_engine_includes_prior_turns_in_conversation_order(tmp_path):
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
    llm = SequentialLLM(["respuesta1", "respuesta2"])
    engine = AgentEngine(
        repo_root=tmp_path,
        llm=llm,
        tools=tools,
        history_store=history_store,
        memory_store=memory,
        output_store=output_store,
    )

    engine.run_event(TriggerEvent(kind="interactive", source="terminal", prompt="prompt 1"))
    engine.run_event(TriggerEvent(kind="interactive", source="terminal", prompt="prompt 2"))

    second_messages = llm.messages_by_call[1]
    # system, prompt1, response1, prompt2
    roles = [item["role"] for item in second_messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert second_messages[1]["content"] == "prompt 1"
    assert second_messages[2]["content"] == "respuesta1"
    assert second_messages[3]["content"] == "prompt 2"


def test_engine_breaks_infinite_same_chunk_stream(tmp_path):
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
    llm = InfiniteSameChunkStreamingLLM()
    engine = AgentEngine(
        repo_root=tmp_path,
        llm=llm,
        tools=tools,
        history_store=history_store,
        memory_store=memory,
        output_store=output_store,
    )
    chunks: list[str] = []
    result = engine.run_event(
        TriggerEvent(kind="interactive", source="test", prompt="hola"),
        on_stream=chunks.append,
    )

    assert llm.stopped is True
    assert result.response == "Hola"
    assert "".join(chunks) == "Hola"
