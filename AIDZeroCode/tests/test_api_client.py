from __future__ import annotations

import json

import pytest

from CORE.api_client import CoreAPIClient, CoreAPIError
from CORE.api_client import RemoteAgentEngine
from CORE.models import TriggerEvent, TurnResult


class _FakeClient:
    def __init__(self) -> None:
        self.simple_calls = 0
        self.stream_calls = 0
        self.reset_calls = 0

    def run_event(self, event: TriggerEvent, *, max_rounds: int = 6) -> TurnResult:
        del event, max_rounds
        self.simple_calls += 1
        return TurnResult(
            event=TriggerEvent(kind="interactive", source="test", prompt="hola"),
            response="respuesta simple",
            rounds=1,
            used_tools=[],
        )

    def stream_run_event(
        self,
        event: TriggerEvent,
        *,
        max_rounds: int = 6,
        on_stream=None,
        on_artifact=None,
    ) -> TurnResult:
        del event, max_rounds
        self.stream_calls += 1
        if on_stream is not None:
            on_stream("respuesta ")
            on_stream("con stream")
        if on_artifact is not None:
            on_artifact({"event": "start", "artifact_id": "r1-a1", "type": "think", "offset": 0})
            on_artifact({"event": "chunk", "artifact_id": "r1-a1", "content": "internal"})
            on_artifact({"event": "end", "artifact_id": "r1-a1"})
        return TurnResult(
            event=TriggerEvent(kind="interactive", source="test", prompt="hola"),
            response="respuesta con stream",
            rounds=1,
            used_tools=["echo"],
        )

    def reset_session(self) -> None:
        self.reset_calls += 1


def test_remote_agent_engine_uses_simple_run_without_callbacks() -> None:
    client = _FakeClient()
    engine = RemoteAgentEngine(client)

    result = engine.run_event(TriggerEvent(kind="interactive", source="test", prompt="hola"))

    assert result.response == "respuesta simple"
    assert client.simple_calls == 1
    assert client.stream_calls == 0


def test_remote_agent_engine_replays_trace_callbacks() -> None:
    client = _FakeClient()
    engine = RemoteAgentEngine(client)
    streamed: list[str] = []
    artifacts: list[dict] = []

    result = engine.run_event(
        TriggerEvent(kind="interactive", source="test", prompt="hola"),
        on_stream=streamed.append,
        on_artifact=artifacts.append,
    )

    assert result.response == "respuesta con stream"
    assert streamed == ["respuesta ", "con stream"]
    assert [item.get("event") for item in artifacts] == ["start", "chunk", "end"]
    assert client.simple_calls == 0
    assert client.stream_calls == 1


def test_remote_agent_engine_reset_session_delegates_to_client() -> None:
    client = _FakeClient()
    engine = RemoteAgentEngine(client)

    engine.reset_session()

    assert client.reset_calls == 1


class _OpenEndedSSE:
    """Iterable that never ends (simulates keep-alive SSE) after seeded lines."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        if self._index < len(self._lines):
            line = self._lines[self._index]
            self._index += 1
            return line
        raise RuntimeError("stream kept open without EOF")


def test_consume_run_event_stream_returns_on_result_without_waiting_eof() -> None:
    event_payload = {
        "event": {
            "kind": "interactive",
            "source": "test",
            "prompt": "hola",
            "created_at": "2026-01-01T00:00:00+00:00",
            "metadata": {},
        },
        "response": "respuesta final",
        "rounds": 1,
        "used_tools": [],
    }
    lines = [
        b"event: stream\n",
        b"data: {\"chunk\": \"hola\"}\n",
        b"\n",
        b"event: result\n",
        ("data: " + json.dumps({"result": event_payload}) + "\n").encode("utf-8"),
        b"\n",
    ]
    streamed: list[str] = []

    result = CoreAPIClient._consume_run_event_stream(
        response=_OpenEndedSSE(lines),
        on_stream=streamed.append,
    )

    assert streamed == ["hola"]
    assert result.response == "respuesta final"


def test_consume_run_event_stream_errors_when_result_missing() -> None:
    lines = [
        b"event: stream\n",
        b"data: {\"chunk\": \"hola\"}\n",
        b"\n",
    ]

    with pytest.raises(CoreAPIError, match="Stream ended without result payload"):
        CoreAPIClient._consume_run_event_stream(response=iter(lines))
