from __future__ import annotations

import urllib.request

from LLMProviders.openai_compatible_provider import OpenAICompatibleProvider


class _FakeResponse:
    def __init__(self, body: str, *, content_type: str) -> None:
        self._body = body
        self.headers = {"Content-Type": content_type}

    def read(self) -> bytes:
        return self._body.encode("utf-8")

    def __iter__(self):
        return iter(())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        return None


def test_stream_generate_text_accepts_non_sse_json(monkeypatch):
    provider = OpenAICompatibleProvider(
        provider_label="test",
        base_url="http://localhost:8080/v1",
        require_api_key=False,
    )

    def _fake_urlopen(request, timeout):  # noqa: ANN001
        del request, timeout
        return _FakeResponse(
            '{"choices":[{"message":{"content":"Respuesta larga completa"}}]}',
            content_type="application/json",
        )

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    chunks = list(provider.stream_generate_text("model", "prompt"))
    assert "".join(chunks) == "Respuesta larga completa"
