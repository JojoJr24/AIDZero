from __future__ import annotations

import json
from threading import Thread
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest

from CORE.models import TriggerEvent, TurnResult
from UI.Android import entrypoint as android_ui


class _DummyHistory:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.enabled = True

    def add_prompt(self, prompt: str) -> list[str]:
        self.prompts.append(prompt)
        return list(self.prompts)


class _DummyEngine:
    def __init__(self) -> None:
        self.events: list[TriggerEvent] = []
        self.reset_calls = 0

    def run_event(self, event: TriggerEvent, *, max_rounds: int = 6, on_stream=None, on_artifact=None) -> TurnResult:
        del max_rounds, on_stream, on_artifact
        self.events.append(event)
        return TurnResult(
            event=event,
            response="android answer",
            rounds=1,
            used_tools=[],
        )

    def reset_session(self) -> None:
        self.reset_calls += 1


def test_android_service_uses_embedded_runtime_history_and_metadata() -> None:
    engine = _DummyEngine()
    history = _DummyHistory()
    service = android_ui.AndroidUIService(
        engine=engine,
        history=history,
        agent_name="default",
    )

    payload = service.handle_prompt(" hola desde android ")

    assert payload["response"] == "android answer"
    assert payload["core_url"] == "embedded"
    assert history.prompts == ["hola desde android"]
    assert len(engine.events) == 1
    event = engine.events[0]
    assert event.kind == "interactive"
    assert event.source == "android"
    assert event.prompt == "hola desde android"
    assert event.metadata == {
        "trigger": "interactive",
        "channel": "android",
        "transport": "web",
    }


def test_android_handler_serves_html_and_chat_endpoint() -> None:
    engine = _DummyEngine()
    history = _DummyHistory()
    service = android_ui.AndroidUIService(
        engine=engine,
        history=history,
        agent_name="default",
    )
    handler = android_ui.build_request_handler(service)
    server = android_ui.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5.0) as response:
            html = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")

        request = Request(
            url=f"http://127.0.0.1:{server.server_port}/api/chat",
            data=json.dumps({"prompt": "hola app"}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5.0)

    assert "text/html" in content_type
    assert "AIDZero on Android" in html
    assert payload["ok"] is True
    assert payload["data"]["response"] == "android answer"
    assert history.prompts == ["hola app"]


def test_android_config_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="app_port must be an integer"):
        android_ui.AndroidUIConfig.from_options({"app_port": "abc"})


def test_run_ui_checks_default_remote_core_before_serving(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, base_url: str) -> None:
            calls["base_url"] = base_url

        def health(self) -> dict[str, str]:
            calls["health_called"] = True
            return {"status": "ok"}

    class _FakeServer:
        def __init__(self, addr, handler) -> None:
            calls["server_addr"] = addr
            calls["handler"] = handler

        def serve_forever(self) -> None:
            calls["served"] = True

        def server_close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(
        android_ui,
        "build_ui_runtime",
        lambda *, repo_root, provider_name, model: SimpleNamespace(
            engine=_DummyEngine(),
            history=_DummyHistory(),
            agent_profile=SimpleNamespace(name="default"),
        ),
    )
    monkeypatch.setattr(android_ui, "CoreAPIClient", _FakeClient)
    monkeypatch.setattr(android_ui, "ThreadingHTTPServer", _FakeServer)

    exit_code = android_ui.run_ui(
        provider_name="openai",
        model="gpt-4o-mini",
        repo_root=tmp_path,
        ui_options={
            "core_url": "http://192.168.1.22:8765",
            "app_host": "0.0.0.0",
            "app_port": "8899",
        },
    )

    assert exit_code == 0
    assert calls["base_url"] == "http://192.168.1.22:8765"
    assert calls["health_called"] is True
    assert calls["server_addr"] == ("0.0.0.0", 8899)
    assert calls["served"] is True
    assert calls["closed"] is True
