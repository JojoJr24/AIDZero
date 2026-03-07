"""HTTP server exposing the core runtime as a separate layer."""

from __future__ import annotations

from argparse import ArgumentParser
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from CORE.agents import AgentProfileManager
from CORE.api_contract import profile_to_dict, trigger_event_from_dict, trigger_event_to_dict, turn_result_to_dict
from CORE.ui_runtime import UIRuntime, build_ui_runtime, profile_disabled_tools
from CORE.tooling import build_default_tool_registry


class CoreRuntimeService:
    """Owns runtime state and handles API actions safely."""

    def __init__(self, *, repo_root: Path, provider_name: str, model: str) -> None:
        self.repo_root = repo_root.resolve()
        self.runtime: UIRuntime = build_ui_runtime(
            repo_root=self.repo_root,
            provider_name=provider_name,
            model=model,
        )
        self._lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        profile = self.runtime.agent_manager.get_active_profile()
        return {
            "status": "ok",
            "provider": self.runtime.engine.llm.provider_name,
            "model": self.runtime.engine.llm.model,
            "active_profile": profile.name,
        }

    def collect_events(self, *, trigger: str, prompt: str | None, consume: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            events = self.runtime.gateway.collect(trigger=trigger, prompt=prompt, consume=consume)
        return [trigger_event_to_dict(event) for event in events]

    def run_event(
        self,
        *,
        event_payload: dict[str, Any],
        max_rounds: int,
        include_trace: bool = False,
    ) -> dict[str, Any]:
        event = trigger_event_from_dict(event_payload)
        artifacts: list[dict[str, Any]] = []
        on_artifact = artifacts.append if include_trace else None
        with self._lock:
            result = self.runtime.engine.run_event(
                event,
                max_rounds=max_rounds,
                on_artifact=on_artifact,
            )
        payload: dict[str, Any] = {"result": turn_result_to_dict(result)}
        if include_trace:
            payload["artifacts"] = artifacts
        return payload

    def run_event_stream(
        self,
        *,
        event_payload: dict[str, Any],
        max_rounds: int,
        on_stream: Callable[[str], None] | None = None,
        on_artifact: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        event = trigger_event_from_dict(event_payload)
        with self._lock:
            result = self.runtime.engine.run_event(
                event,
                max_rounds=max_rounds,
                on_stream=on_stream,
                on_artifact=on_artifact,
            )
        return turn_result_to_dict(result)

    def reset_session(self) -> None:
        with self._lock:
            reset = getattr(self.runtime.engine, "reset_session", None)
            if callable(reset):
                reset()

    def history_add(self, *, prompt: str) -> list[str]:
        with self._lock:
            return self.runtime.history.add_prompt(prompt)

    def history_list(self, *, limit: int | None) -> list[str]:
        with self._lock:
            return self.runtime.history.list_prompts(limit=limit)

    def list_profiles(self) -> list[str]:
        with self._lock:
            return self.runtime.agent_manager.list_profile_names()

    def get_active_profile(self) -> dict[str, Any]:
        with self._lock:
            profile = self.runtime.agent_manager.get_active_profile()
        return profile_to_dict(profile)

    def activate_profile(self, *, name: str) -> dict[str, Any]:
        with self._lock:
            profile = self.runtime.agent_manager.set_active_profile(name)
            self._apply_profile(profile)
        return profile_to_dict(profile)

    def _apply_profile(self, profile) -> None:
        tools = build_default_tool_registry(
            self.repo_root,
            self.runtime.engine.memory_store,
            enabled_names=profile.enabled_tools,
            disabled_names=profile_disabled_tools(profile),
        )
        self.runtime.engine.tools = tools
        self.runtime.engine.system_prompt_override = profile.system_prompt
        self.runtime.engine.memory_store.enabled = profile.memory_enabled
        self.runtime.engine.history_enabled = profile.history_enabled
        self.runtime.history.enabled = profile.history_enabled


class _CoreRequestHandler(BaseHTTPRequestHandler):
    service: CoreRuntimeService

    server_version = "AIDZeroCoreAPI/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        del format, args

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            return self._ok(self.service.health())

        if parsed.path == "/history/list":
            query = parse_qs(parsed.query)
            raw_limit = query.get("limit", [None])[0]
            limit = _to_optional_int(raw_limit)
            prompts = self.service.history_list(limit=limit)
            return self._ok({"prompts": prompts})

        if parsed.path == "/agent/profiles":
            return self._ok({"names": self.service.list_profiles()})

        if parsed.path == "/agent/active":
            return self._ok({"profile": self.service.get_active_profile()})

        return self._error(404, f"Unknown endpoint: {parsed.path}")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/engine/run_event_stream":
            return self._handle_run_event_stream()
        if parsed.path == "/engine/session/reset":
            payload = self._read_json_body()
            if payload is None:
                return
            self.service.reset_session()
            return self._ok({"status": "ok"})
        payload = self._read_json_body()
        if payload is None:
            return

        if parsed.path == "/gateway/collect":
            trigger = str(payload.get("trigger", "interactive"))
            prompt = payload.get("prompt")
            if prompt is not None:
                prompt = str(prompt)
            consume = bool(payload.get("consume", True))
            events = self.service.collect_events(trigger=trigger, prompt=prompt, consume=consume)
            return self._ok({"events": events})

        if parsed.path == "/engine/run_event":
            event_payload = payload.get("event")
            if not isinstance(event_payload, dict):
                return self._error(400, "'event' must be a JSON object")
            max_rounds = _to_int(payload.get("max_rounds"), default=6, minimum=1)
            include_trace = bool(payload.get("include_trace", False))
            try:
                result_payload = self.service.run_event(
                    event_payload=event_payload,
                    max_rounds=max_rounds,
                    include_trace=include_trace,
                )
            except Exception as error:  # noqa: BLE001
                return self._error(502, f"Engine execution failed: {error}")
            return self._ok(result_payload)

        if parsed.path == "/history/add":
            prompt = str(payload.get("prompt", ""))
            prompts = self.service.history_add(prompt=prompt)
            return self._ok({"prompts": prompts})

        if parsed.path == "/agent/activate":
            name = str(payload.get("name", "")).strip()
            if not name:
                return self._error(400, "'name' cannot be empty")
            profile = self.service.activate_profile(name=name)
            return self._ok({"profile": profile})

        return self._error(404, f"Unknown endpoint: {parsed.path}")

    def _handle_run_event_stream(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        event_payload = payload.get("event")
        if not isinstance(event_payload, dict):
            return self._error(400, "'event' must be a JSON object")
        max_rounds = _to_int(payload.get("max_rounds"), default=6, minimum=1)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def _send_sse(event_name: str, data: dict[str, Any]) -> None:
            body = (
                f"event: {event_name}\n"
                f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            ).encode("utf-8")
            self.wfile.write(body)
            self.wfile.flush()

        try:
            result_payload = self.service.run_event_stream(
                event_payload=event_payload,
                max_rounds=max_rounds,
                on_stream=lambda chunk: _send_sse("stream", {"chunk": str(chunk)}),
                on_artifact=lambda artifact: _send_sse("artifact", dict(artifact)),
            )
            _send_sse("result", {"result": result_payload})
        except Exception as error:  # noqa: BLE001
            _send_sse("error", {"message": f"Engine execution failed: {error}"})
        finally:
            self.close_connection = True

    def _read_json_body(self) -> dict[str, Any] | None:
        raw_length = self.headers.get("Content-Length")
        length = _to_int(raw_length, default=0, minimum=0)
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as error:
            self._error(400, f"Invalid JSON body: {error}")
            return None
        if not isinstance(payload, dict):
            self._error(400, "JSON body must be an object")
            return None
        return payload

    def _ok(self, data: dict[str, Any]) -> None:
        self._write(200, {"ok": True, "data": data})

    def _error(self, status: int, message: str) -> None:
        self._write(status, {"ok": False, "error": message})

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _to_int(raw: Any, *, default: int, minimum: int | None = None) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _to_optional_int(raw: Any) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return _to_int(text, default=0, minimum=0)


def serve_core_api(*, repo_root: Path, provider_name: str, model: str, host: str, port: int) -> int:
    service = CoreRuntimeService(repo_root=repo_root, provider_name=provider_name, model=model)

    class Handler(_CoreRequestHandler):
        pass

    Handler.service = service
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            raise RuntimeError(
                f"No se pudo iniciar el core API: el puerto {port} ya está en uso en {host}."
            ) from error
        raise RuntimeError(f"No se pudo iniciar el core API en {host}:{port}: {error}") from error
    print(f"AIDZero core API listening on http://{host}:{port}")
    print(f"Provider={provider_name} Model={model}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _parse_args() -> Any:
    parser = ArgumentParser(description="Run AIDZero core layer as HTTP API")
    parser.add_argument("--agent", default=None, help="Agent profile name from Agents/<name>/<name>.json")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host/IP")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root path",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()

    profile_manager = AgentProfileManager(repo_root)
    if args.agent and str(args.agent).strip():
        try:
            profile = profile_manager.set_active_profile(str(args.agent).strip())
        except Exception as error:  # noqa: BLE001
            print(f"error> {error}")
            return 2
    else:
        profile = profile_manager.get_active_profile()

    try:
        return serve_core_api(
            repo_root=repo_root,
            provider_name=profile.runtime_provider,
            model=profile.runtime_model,
            host=str(args.host),
            port=int(args.port),
        )
    except RuntimeError as error:
        print(f"error> {error}")
        print("tip> Cerrá el proceso que usa ese puerto o ejecutá con --port 8766 (u otro libre).")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
