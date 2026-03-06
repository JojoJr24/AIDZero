"""WhatsApp webhook UI entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from typing import Any
from urllib.parse import parse_qs
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CORE.api_client import (
    CoreAPIClient,
    RemoteAgentEngine,
    RemoteAgentProfileManager,
    RemotePromptHistoryStore,
)
from CORE.models import TriggerEvent
from CORE.ui_runtime import build_ui_runtime


@dataclass(frozen=True)
class WhatsAppUIConfig:
    webhook_host: str
    webhook_port: int
    webhook_path: str
    response_format: str

    @classmethod
    def from_options(cls, options: dict[str, str] | None) -> "WhatsAppUIConfig":
        raw_options = options or {}
        webhook_host = str(raw_options.get("webhook_host", "0.0.0.0")).strip() or "0.0.0.0"
        webhook_port = _parse_port(raw_options.get("webhook_port"), default=8780)
        webhook_path = _normalize_path(str(raw_options.get("webhook_path", "/webhook")))
        response_format = str(raw_options.get("response_format", "twiml")).strip().lower() or "twiml"
        if response_format not in {"twiml", "json"}:
            raise ValueError("response_format must be 'twiml' or 'json'.")
        return cls(
            webhook_host=webhook_host,
            webhook_port=webhook_port,
            webhook_path=webhook_path,
            response_format=response_format,
        )


@dataclass(frozen=True)
class IncomingWhatsAppMessage:
    prompt: str
    metadata: dict[str, Any]
    transport: str


class WhatsAppWebhookService:
    """Owns a single chat session bridged from WhatsApp into the runtime."""

    def __init__(
        self,
        *,
        engine,
        history,
        agent_name: str,
        response_format: str,
        core_url: str = "",
    ) -> None:
        self.engine = engine
        self.history = history
        self.agent_name = agent_name.strip() or "unknown"
        self.response_format = response_format
        self.core_url = core_url.strip()
        self._lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "agent": self.agent_name,
            "response_format": self.response_format,
            "core_url": self.core_url or "embedded",
        }

    def handle_prompt(self, prompt: str, *, metadata: dict[str, Any] | None = None) -> str:
        text = prompt.strip()
        if not text:
            return "Please send a non-empty text message."

        command_reply = self._handle_command(text)
        if command_reply is not None:
            return command_reply

        event_metadata = {
            "trigger": "interactive",
            "channel": "whatsapp",
            **(metadata or {}),
        }
        event = TriggerEvent(
            kind="interactive",
            source="whatsapp",
            prompt=text,
            metadata=event_metadata,
        )

        with self._lock:
            self.history.add_prompt(text)
            result = self.engine.run_event(event)
        response = result.response.strip()
        return response or "No response generated."

    def _handle_command(self, prompt: str) -> str | None:
        normalized = prompt.strip().lower()
        if normalized not in {"/new", "/reset"}:
            return None

        reset = getattr(self.engine, "reset_session", None)
        if not callable(reset):
            return "This runtime does not support session reset."

        with self._lock:
            reset()
        return "Started a new conversation."


def build_request_handler(
    service: WhatsAppWebhookService,
    *,
    webhook_path: str,
) -> type[BaseHTTPRequestHandler]:
    class _WhatsAppRequestHandler(BaseHTTPRequestHandler):
        server_version = "AIDZeroWhatsappUI/1.0"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            del format, args

        def do_GET(self) -> None:  # noqa: N802
            if _normalize_path(self.path) != "/health":
                self._write_json(404, {"ok": False, "error": f"Unknown endpoint: {self.path}"})
                return
            self._write_json(200, {"ok": True, "data": service.health()})

        def do_POST(self) -> None:  # noqa: N802
            if _normalize_path(self.path) != webhook_path:
                self._write_json(404, {"ok": False, "error": f"Unknown endpoint: {self.path}"})
                return

            raw_body = self._read_body()
            content_type = str(self.headers.get("Content-Type", ""))
            wants_twiml = service.response_format == "twiml" or _looks_like_twilio(content_type)

            try:
                incoming = parse_incoming_whatsapp_message(
                    raw_body=raw_body,
                    content_type=content_type,
                )
                reply_text = service.handle_prompt(incoming.prompt, metadata=incoming.metadata)
                self._write_message_reply(reply_text, wants_twiml=wants_twiml)
            except ValueError as error:
                self._write_message_reply(str(error), wants_twiml=wants_twiml, status=200 if wants_twiml else 400)
            except Exception as error:  # noqa: BLE001
                self._write_message_reply(f"Provider/core error: {error}", wants_twiml=wants_twiml)

        def _read_body(self) -> bytes:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = max(0, int(raw_length))
            except (TypeError, ValueError):
                length = 0
            return self.rfile.read(length) if length > 0 else b""

        def _write_message_reply(self, message: str, *, wants_twiml: bool, status: int = 200) -> None:
            text = message.strip() or "No response generated."
            if wants_twiml:
                body = render_twiml_message(text)
                self.send_response(status)
                self.send_header("Content-Type", "text/xml; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            payload = {"ok": status < 400, "response": text}
            self._write_json(status, payload)

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return _WhatsAppRequestHandler


def parse_incoming_whatsapp_message(
    *,
    raw_body: bytes,
    content_type: str,
) -> IncomingWhatsAppMessage:
    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type == "application/json":
        return _parse_json_message(raw_body)
    if normalized_content_type in {"application/x-www-form-urlencoded", ""}:
        return _parse_twilio_form_message(raw_body)
    raise ValueError(f"Unsupported content type: {normalized_content_type or 'unknown'}")


def render_twiml_message(message: str) -> bytes:
    safe_text = escape(message, {"'": "&apos;", '"': "&quot;"})
    payload = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{safe_text}</Message></Response>'
    return payload.encode("utf-8")


def run_ui(
    *,
    provider_name: str,
    model: str,
    user_request: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    yes: bool = False,
    repo_root: Path | None = None,
    ui_options: dict[str, str] | None = None,
) -> int:
    del dry_run, overwrite, yes

    root = (repo_root or REPO_ROOT).resolve()
    try:
        config = WhatsAppUIConfig.from_options(ui_options)
    except ValueError as error:
        print(f"error> {error}")
        return 2

    options = ui_options or {}
    core_url = str(options.get("core_url", "")).strip()

    if core_url:
        client = CoreAPIClient(core_url)
        client.health()
        engine = RemoteAgentEngine(client)
        history = RemotePromptHistoryStore(client)
        agent_manager = RemoteAgentProfileManager(client, repo_root=root)
        agent_profile = agent_manager.get_active_profile()
    else:
        runtime = build_ui_runtime(repo_root=root, provider_name=provider_name, model=model)
        engine = runtime.engine
        history = runtime.history
        agent_profile = runtime.agent_profile

    if user_request and user_request.strip():
        print("warning> --request is ignored by the WhatsApp UI. Send the prompt from WhatsApp instead.")

    service = WhatsAppWebhookService(
        engine=engine,
        history=history,
        agent_name=agent_profile.name,
        response_format=config.response_format,
        core_url=core_url,
    )
    handler = build_request_handler(service, webhook_path=config.webhook_path)

    try:
        server = ThreadingHTTPServer((config.webhook_host, config.webhook_port), handler)
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            print(
                "error> "
                f"Cannot start WhatsApp UI: port {config.webhook_port} is already in use on {config.webhook_host}."
            )
            return 2
        print(f"error> Cannot start WhatsApp UI on {config.webhook_host}:{config.webhook_port}: {error}")
        return 2

    print("WhatsApp UI listening:")
    print(f"- webhook: http://{config.webhook_host}:{config.webhook_port}{config.webhook_path}")
    print(f"- agent: {agent_profile.name}")
    print(f"- response_format: {config.response_format}")
    if core_url:
        print(f"- core_url: {core_url}")
    print("- commands: /new, /reset")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _looks_like_twilio(content_type: str) -> bool:
    return "application/x-www-form-urlencoded" in content_type.lower()


def _normalize_path(raw_path: str) -> str:
    text = (raw_path or "").strip()
    if not text:
        return "/"
    base = text.split("?", 1)[0]
    normalized = "/" + base.lstrip("/")
    return normalized.rstrip("/") or "/"


def _parse_port(raw_value: Any, *, default: int) -> int:
    if raw_value is None:
        return default
    try:
        port = int(str(raw_value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError("webhook_port must be an integer.") from error
    if not 1 <= port <= 65535:
        raise ValueError("webhook_port must be between 1 and 65535.")
    return port


def _parse_twilio_form_message(raw_body: bytes) -> IncomingWhatsAppMessage:
    try:
        payload = parse_qs(raw_body.decode("utf-8"), keep_blank_values=True)
    except UnicodeDecodeError as error:
        raise ValueError(f"Invalid form body encoding: {error}") from error

    prompt = _first_value(payload, "Body", "body").strip()
    if not prompt:
        raise ValueError("Please send a text message to the WhatsApp number.")

    metadata = {
        "transport": "twilio",
        "from": _first_value(payload, "From", "from"),
        "wa_id": _first_value(payload, "WaId", "wa_id"),
        "profile_name": _first_value(payload, "ProfileName", "profile_name"),
        "message_sid": _first_value(payload, "MessageSid", "SmsMessageSid", "message_sid"),
    }
    return IncomingWhatsAppMessage(prompt=prompt, metadata=metadata, transport="twilio")


def _parse_json_message(raw_body: bytes) -> IncomingWhatsAppMessage:
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON body: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")

    direct_prompt = _first_non_empty_string(payload, "prompt", "text", "body", "message")
    if direct_prompt:
        metadata = {
            "transport": "json",
            "from": _first_non_empty_string(payload, "from", "sender"),
        }
        return IncomingWhatsAppMessage(prompt=direct_prompt, metadata=metadata, transport="json")

    meta_message = _extract_meta_message(payload)
    if meta_message is not None:
        return meta_message
    raise ValueError("Unsupported JSON payload for WhatsApp UI.")


def _extract_meta_message(payload: dict[str, Any]) -> IncomingWhatsAppMessage | None:
    entries = payload.get("entry")
    if not isinstance(entries, list):
        return None

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        changes = entry.get("changes")
        if not isinstance(changes, list):
            continue
        for change in changes:
            if not isinstance(change, dict):
                continue
            value = change.get("value")
            if not isinstance(value, dict):
                continue
            messages = value.get("messages")
            if not isinstance(messages, list):
                continue
            contacts = value.get("contacts")
            wa_id = ""
            if isinstance(contacts, list) and contacts:
                first_contact = contacts[0]
                if isinstance(first_contact, dict):
                    wa_id = str(first_contact.get("wa_id", "")).strip()
            for message in messages:
                if not isinstance(message, dict):
                    continue
                text_payload = message.get("text")
                if not isinstance(text_payload, dict):
                    continue
                body = str(text_payload.get("body", "")).strip()
                if not body:
                    continue
                metadata = {
                    "transport": "meta",
                    "from": str(message.get("from", "")).strip() or wa_id,
                    "wa_id": wa_id,
                    "message_id": str(message.get("id", "")).strip(),
                }
                return IncomingWhatsAppMessage(prompt=body, metadata=metadata, transport="meta")
    return None


def _first_non_empty_string(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return ""


def _first_value(payload: dict[str, list[str]], *keys: str) -> str:
    for key in keys:
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value).strip()
            if text:
                return text
    return ""
