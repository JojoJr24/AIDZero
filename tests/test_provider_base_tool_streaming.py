from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import pytest

from agent.models import ComponentCatalog
from agent.provider_base import ToolSpec, ProviderBaseRuntime, _extract_stream_text, _try_extract_tool_call


class FakeStreamingProvider:
    def __init__(
        self,
        rounds: list[list[dict[str, Any]]],
        *,
        chat_responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self._rounds = rounds
        self._chat_responses = chat_responses or []
        self.calls: list[list[dict[str, Any]]] = []
        self.chat_calls: list[list[dict[str, Any]]] = []
        self.stop_stream_calls = 0

    def stream_chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any):
        del model, kwargs
        self.calls.append([dict(message) for message in messages])
        if not self._rounds:
            return iter(())
        return iter(self._rounds.pop(0))

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        del model, kwargs
        self.chat_calls.append([dict(message) for message in messages])
        if not self._chat_responses:
            return {"choices": [{"message": {"content": ""}}]}
        return self._chat_responses.pop(0)

    def stop_stream(self) -> None:
        self.stop_stream_calls += 1


def _catalog() -> ComponentCatalog:
    return ComponentCatalog(
        root=Path.cwd(),
        llm_providers=[],
        skills=[],
        tools=[],
        mcp=[],
        ui=[],
    )


def _openai_stream_event(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


def test_try_extract_tool_call_from_fenced_block() -> None:
    payload = "prefix\n<AID_TOOL_CALL>{\"name\":\"aid_list_skills\",\"arguments\":{}}</AID_TOOL_CALL>\n"
    parsed = _try_extract_tool_call(payload, allowed_tool_names={"aid_list_skills"})
    assert parsed is not None
    assert parsed["name"] == "aid_list_skills"
    assert parsed["arguments"] == {}


def test_extract_stream_text_supports_multiple_provider_shapes() -> None:
    openai_chunks = _extract_stream_text({"choices": [{"delta": {"content": "hello"}}]})
    assert openai_chunks == ["hello"]

    claude_chunks = _extract_stream_text(
        {"type": "content_block_delta", "delta": {"text": "world"}}
    )
    assert claude_chunks == ["world"]

    gemini_chunks = _extract_stream_text(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "from"},
                            {"text": "gemini"},
                        ]
                    }
                }
            ]
        }
    )
    assert gemini_chunks == ["from", "gemini"]


def test_runtime_executes_tool_and_continues_after_response_message() -> None:
    provider = FakeStreamingProvider(
        rounds=[
            [
                _openai_stream_event("Plan: "),
                _openai_stream_event("<AID_TOOL_CALL>\n"),
                _openai_stream_event('{"name":"aid_list_skills","arguments":{}}'),
                _openai_stream_event("\n</AID_TOOL_CALL>"),
                _openai_stream_event("this should not be consumed"),
            ],
            [_openai_stream_event("Done.")],
        ]
    )
    runtime = ProviderBaseRuntime(
        provider=provider,
        model="fake-model",
        repo_root=Path.cwd(),
        catalog=_catalog(),
    )

    answer = runtime.ask(prompt="test prompt", ui_name="terminal")

    assert answer == "Plan: Done."
    assert len(provider.calls) == 2
    assert provider.stop_stream_calls == 1
    second_round_messages = provider.calls[1]
    assert any(message.get("role") == "user" and "Response:" in str(message.get("content")) for message in second_round_messages)


def test_runtime_emits_llm_invocation_trace_per_round() -> None:
    provider = FakeStreamingProvider(
        rounds=[
            [
                _openai_stream_event("Plan: "),
                _openai_stream_event("<AID_TOOL_CALL>\n"),
                _openai_stream_event('{"name":"aid_list_skills","arguments":{}}'),
                _openai_stream_event("\n</AID_TOOL_CALL>"),
            ],
            [_openai_stream_event("Done.")],
        ]
    )
    runtime = ProviderBaseRuntime(
        provider=provider,
        model="fake-model",
        repo_root=Path.cwd(),
        catalog=_catalog(),
    )
    traces: list[dict[str, Any]] = []

    answer = runtime.ask(
        prompt="test prompt",
        ui_name="terminal",
        invocation_tracer=lambda payload: traces.append(payload),
    )

    assert answer == "Plan: Done."
    assert len(traces) == 2
    assert traces[0]["round"] == 1
    assert traces[0]["tool_call"]["name"] == "aid_list_skills"
    assert traces[0]["finished"] is False
    assert traces[1]["round"] == 2
    assert traces[1]["finished"] is True


def test_try_extract_tool_call_ignores_unknown_tool_name() -> None:
    payload = '<AID_TOOL_CALL>{"name":"unknown_tool","arguments":{}}</AID_TOOL_CALL>'
    parsed = _try_extract_tool_call(payload, allowed_tool_names={"aid_list_skills"})
    assert parsed is None


def test_runtime_raises_when_model_never_completes_after_tool_output() -> None:
    provider = FakeStreamingProvider(
        rounds=[
            [
                _openai_stream_event("Plan: "),
                _openai_stream_event("<AID_TOOL_CALL>\n"),
                _openai_stream_event('{"name":"aid_list_skills","arguments":{}}'),
                _openai_stream_event("\n</AID_TOOL_CALL>"),
            ],
            [],
        ]
    )
    runtime = ProviderBaseRuntime(
        provider=provider,
        model="fake-model",
        repo_root=Path.cwd(),
        catalog=_catalog(),
    )

    with pytest.raises(RuntimeError, match="did not provide a final response"):
        runtime.ask(prompt="test prompt", ui_name="terminal")
    assert len(provider.chat_calls) >= 1


def test_runtime_uses_non_streaming_fallback_when_post_tool_stream_is_empty() -> None:
    provider = FakeStreamingProvider(
        rounds=[
            [
                _openai_stream_event("Plan: "),
                _openai_stream_event("<AID_TOOL_CALL>\n"),
                _openai_stream_event('{"name":"aid_list_skills","arguments":{}}'),
                _openai_stream_event("\n</AID_TOOL_CALL>"),
            ],
            [],
        ],
        chat_responses=[{"choices": [{"message": {"content": "Done."}}]}],
    )
    runtime = ProviderBaseRuntime(
        provider=provider,
        model="fake-model",
        repo_root=Path.cwd(),
        catalog=_catalog(),
    )

    answer = runtime.ask(prompt="test prompt", ui_name="terminal")

    assert answer == "Plan: Done."
    assert len(provider.calls) == 2
    assert len(provider.chat_calls) == 1


def test_runtime_injects_tool_error_into_context_and_continues() -> None:
    provider = FakeStreamingProvider(
        rounds=[
            [
                _openai_stream_event("Checking skill. "),
                _openai_stream_event("<AID_TOOL_CALL>\n"),
                _openai_stream_event('{"name":"aid_read_skill","arguments":{}}'),
                _openai_stream_event("\n</AID_TOOL_CALL>"),
            ],
            [_openai_stream_event("I could not read that skill without a name.")],
        ]
    )
    runtime = ProviderBaseRuntime(
        provider=provider,
        model="fake-model",
        repo_root=Path.cwd(),
        catalog=_catalog(),
    )

    answer = runtime.ask(prompt="test prompt", ui_name="terminal")

    assert answer == "Checking skill. I could not read that skill without a name."
    second_round_messages = provider.calls[1]
    response_messages = [
        message for message in second_round_messages if message.get("role") == "user" and "Response:" in str(message.get("content"))
    ]
    assert response_messages
    serialized = str(response_messages[-1].get("content"))
    assert '"status": "error"' in serialized
    assert '"tool_name": "aid_read_skill"' in serialized


def test_runtime_parses_mcp_gateway_tool_call_and_continues_after_tool_response() -> None:
    provider = FakeStreamingProvider(
        rounds=[
            [
                _openai_stream_event("Buscando. "),
                _openai_stream_event("<AID_TOOL_CALL>\n"),
                _openai_stream_event(
                    '{"name":"aid_mcp_gateway_call","arguments":{"tool":"tool_search","payload":{"query":"Patata","limit":2}}}'
                ),
                _openai_stream_event("\n</AID_TOOL_CALL>"),
            ],
            [_openai_stream_event("Terminado.")],
        ]
    )
    runtime = ProviderBaseRuntime(
        provider=provider,
        model="fake-model",
        repo_root=Path.cwd(),
        catalog=_catalog(),
    )
    captured_arguments: list[dict[str, Any]] = []
    mcp_spec = runtime._tool_specs["aid_mcp_gateway_call"]
    runtime._tool_specs["aid_mcp_gateway_call"] = ToolSpec(
        definition=mcp_spec.definition,
        execute=lambda args: captured_arguments.append(args) or {"isError": False, "matches": []},
    )

    answer = runtime.ask(prompt="test prompt", ui_name="terminal")

    assert answer == "Buscando. Terminado."
    assert captured_arguments == [{"tool": "tool_search", "payload": {"query": "Patata", "limit": 2}}]
    assert provider.stop_stream_calls == 1


def test_mcp_gateway_tool_timeout_returns_error_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeStreamingProvider(rounds=[[]])
    runtime = ProviderBaseRuntime(
        provider=provider,
        model="fake-model",
        repo_root=Path.cwd(),
        catalog=_catalog(),
    )

    def _raise_timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=["node"], timeout=45)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    payload = runtime._tool_mcp_gateway_call({"tool": "tool_search", "payload": {"query": "Patata"}})

    assert isinstance(payload, dict)
    assert "timed out" in str(payload.get("error", "")).lower()
