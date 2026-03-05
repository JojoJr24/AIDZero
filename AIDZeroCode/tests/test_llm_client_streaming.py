from __future__ import annotations

from CORE.llm_client import LLMClient


class _ProviderPrefersStreamChat:
    def stream_chat(self, model, messages, **kwargs):
        del model, messages, kwargs
        yield {"choices": [{"delta": {"content": "Hola"}}]}
        yield {"choices": [{"delta": {"content": " mundo"}}]}

    def stream_generate_text(self, model, prompt, **kwargs):
        del model, prompt, kwargs
        raise AssertionError("stream_generate_text should not be used when stream_chat exists")


class _ProviderMessageContentEvent:
    def stream_chat(self, model, messages, **kwargs):
        del model, messages, kwargs
        yield {"choices": [{"message": {"content": "Respuesta completa"}}]}


class _ProviderNonStreamingToolCallOnly:
    def chat(self, model, messages, **kwargs):
        del model, messages, kwargs
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": "{\"value\":\"ok\"}",
                                },
                            }
                        ],
                    }
                }
            ]
        }


class _ProviderStreamingToolCallOnly:
    def stream_chat(self, model, messages, **kwargs):
        del model, messages, kwargs
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "echo", "arguments": "{\"value\":\""},
                            }
                        ]
                    }
                }
            ]
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": "ok\"}"},
                            }
                        ]
                    }
                }
            ]
        }


def _build_client(provider):
    client = LLMClient.__new__(LLMClient)
    client.provider = provider
    client.model = "test-model"
    client.provider_name = "test-provider"
    return client


def test_complete_stream_prefers_stream_chat():
    client = _build_client(_ProviderPrefersStreamChat())
    chunks = list(client.complete_stream([{"role": "user", "content": "hola"}]))
    assert "".join(chunks) == "Hola mundo"


def test_complete_stream_accepts_message_content_events():
    client = _build_client(_ProviderMessageContentEvent())
    chunks = list(client.complete_stream([{"role": "user", "content": "hola"}]))
    assert "".join(chunks) == "Respuesta completa"


def test_complete_serializes_native_tool_call_to_tool_call_block():
    client = _build_client(_ProviderNonStreamingToolCallOnly())
    text = client.complete([{"role": "user", "content": "hola"}])
    assert text == '<tool_call>{"name": "echo", "arguments": {"value": "ok"}}</tool_call>'


def test_complete_stream_serializes_native_tool_call_when_no_text():
    client = _build_client(_ProviderStreamingToolCallOnly())
    chunks = list(client.complete_stream([{"role": "user", "content": "hola"}]))
    assert "".join(chunks) == '<tool_call>{"name": "echo", "arguments": {"value": "ok"}}</tool_call>'
