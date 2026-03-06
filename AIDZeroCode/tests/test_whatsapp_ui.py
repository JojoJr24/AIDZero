from __future__ import annotations

import json
from threading import Thread
from types import SimpleNamespace
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from CORE.models import TriggerEvent, TurnResult
from UI.Whatsapp import entrypoint as whatsapp_ui


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
            response="answer <done>",
            rounds=1,
            used_tools=[],
        )

    def reset_session(self) -> None:
        self.reset_calls += 1


def test_parse_incoming_whatsapp_message_from_twilio_form() -> None:
    incoming = whatsapp_ui.parse_incoming_whatsapp_message(
        raw_body=urlencode(
            {
                "Body": " hola desde whatsapp ",
                "From": "whatsapp:+5491112345678",
                "ProfileName": "Tester",
                "MessageSid": "SM123",
            }
        ).encode("utf-8"),
        content_type="application/x-www-form-urlencoded",
    )

    assert incoming.prompt == "hola desde whatsapp"
    assert incoming.metadata["from"] == "whatsapp:+5491112345678"
    assert incoming.metadata["profile_name"] == "Tester"
    assert incoming.metadata["message_sid"] == "SM123"
    assert incoming.transport == "twilio"


def test_parse_incoming_whatsapp_message_from_meta_json() -> None:
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "contacts": [{"wa_id": "5491112345678"}],
                            "messages": [
                                {
                                    "from": "5491112345678",
                                    "id": "wamid.abc",
                                    "text": {"body": "hola meta"},
                                }
                            ],
                        }
                    }
                ]
            }
        ]
    }

    incoming = whatsapp_ui.parse_incoming_whatsapp_message(
        raw_body=json.dumps(payload).encode("utf-8"),
        content_type="application/json",
    )

    assert incoming.prompt == "hola meta"
    assert incoming.metadata["from"] == "5491112345678"
    assert incoming.metadata["message_id"] == "wamid.abc"
    assert incoming.transport == "meta"


def test_whatsapp_service_supports_session_reset_command() -> None:
    engine = _DummyEngine()
    history = _DummyHistory()
    service = whatsapp_ui.WhatsAppWebhookService(
        engine=engine,
        history=history,
        agent_name="default",
        response_format="twiml",
    )

    reply = service.handle_prompt("/new")

    assert reply == "Started a new conversation."
    assert engine.reset_calls == 1
    assert history.prompts == []
    assert engine.events == []


def test_whatsapp_handler_returns_twiml_and_forwards_event() -> None:
    engine = _DummyEngine()
    history = _DummyHistory()
    service = whatsapp_ui.WhatsAppWebhookService(
        engine=engine,
        history=history,
        agent_name="default",
        response_format="twiml",
    )
    handler = whatsapp_ui.build_request_handler(service, webhook_path="/webhook")
    server = whatsapp_ui.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        request = Request(
            url=f"http://127.0.0.1:{server.server_port}/webhook",
            data=urlencode(
                {
                    "Body": "hola runtime",
                    "From": "whatsapp:+5491112345678",
                }
            ).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urlopen(request, timeout=5.0) as response:
            body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5.0)

    assert "text/xml" in content_type
    assert "<Message>answer &lt;done&gt;</Message>" in body
    assert history.prompts == ["hola runtime"]
    assert len(engine.events) == 1
    event = engine.events[0]
    assert event.kind == "interactive"
    assert event.source == "whatsapp"
    assert event.prompt == "hola runtime"
    assert event.metadata == {
        "trigger": "interactive",
        "channel": "whatsapp",
        "transport": "twilio",
        "from": "whatsapp:+5491112345678",
        "wa_id": "",
        "profile_name": "",
        "message_sid": "",
    }


def test_run_ui_uses_remote_core_healthcheck(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, base_url: str) -> None:
            calls["base_url"] = base_url

        def health(self) -> dict[str, str]:
            calls["health_called"] = True
            return {"status": "ok"}

    class _FakeRemoteEngine:
        def __init__(self, client) -> None:
            calls["engine_client"] = client

    class _FakeRemoteHistory:
        def __init__(self, client) -> None:
            calls["history_client"] = client

    class _FakeProfileManager:
        def __init__(self, client, *, repo_root) -> None:
            calls["profile_client"] = client
            calls["repo_root"] = repo_root

        def get_active_profile(self):
            return SimpleNamespace(name="default")

    class _FakeServer:
        def __init__(self, addr, handler) -> None:
            calls["server_addr"] = addr
            calls["handler"] = handler

        def serve_forever(self) -> None:
            calls["served"] = True

        def server_close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(whatsapp_ui, "CoreAPIClient", _FakeClient)
    monkeypatch.setattr(whatsapp_ui, "RemoteAgentEngine", _FakeRemoteEngine)
    monkeypatch.setattr(whatsapp_ui, "RemotePromptHistoryStore", _FakeRemoteHistory)
    monkeypatch.setattr(whatsapp_ui, "RemoteAgentProfileManager", _FakeProfileManager)
    monkeypatch.setattr(whatsapp_ui, "ThreadingHTTPServer", _FakeServer)

    exit_code = whatsapp_ui.run_ui(
        provider_name="openai",
        model="gpt-4o-mini",
        repo_root=tmp_path,
        ui_options={
            "core_url": "http://127.0.0.1:8765",
            "webhook_host": "127.0.0.1",
            "webhook_port": "8877",
        },
    )

    assert exit_code == 0
    assert calls["base_url"] == "http://127.0.0.1:8765"
    assert calls["health_called"] is True
    assert calls["server_addr"] == ("127.0.0.1", 8877)
    assert calls["served"] is True
    assert calls["closed"] is True
