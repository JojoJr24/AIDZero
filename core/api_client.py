"""HTTP client + local proxies to run UI against a remote core layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from core.agents import AgentProfile
from core.api_contract import profile_from_dict, trigger_event_from_dict, turn_result_from_dict
from core.models import TriggerEvent, TurnResult


class CoreAPIError(RuntimeError):
    """Raised when the core API returns an error or invalid payload."""


class CoreAPIClient:
    """Minimal JSON client for the split core runtime server."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 180.0) -> None:
        normalized = base_url.strip()
        if not normalized:
            raise ValueError("core_url cannot be empty")
        if not normalized.startswith(("http://", "https://")):
            normalized = f"http://{normalized}"
        self.base_url = normalized.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def collect_events(self, *, trigger: str, prompt: str | None, consume: bool = True) -> list[TriggerEvent]:
        data = self._request(
            "POST",
            "/gateway/collect",
            payload={"trigger": trigger, "prompt": prompt, "consume": consume},
        )
        rows = data.get("events")
        if not isinstance(rows, list):
            return []
        out: list[TriggerEvent] = []
        for row in rows:
            if isinstance(row, dict):
                out.append(trigger_event_from_dict(row))
        return out

    def run_event(self, event: TriggerEvent, *, max_rounds: int = 6) -> TurnResult:
        data = self._request(
            "POST",
            "/engine/run_event",
            payload={
                "event": {
                    "kind": event.kind,
                    "source": event.source,
                    "prompt": event.prompt,
                    "created_at": event.created_at,
                    "metadata": dict(event.metadata),
                },
                "max_rounds": max_rounds,
            },
        )
        result_payload = data.get("result")
        if not isinstance(result_payload, dict):
            raise CoreAPIError("Missing result payload from core API")
        return turn_result_from_dict(result_payload)

    def run_event_with_trace(
        self,
        event: TriggerEvent,
        *,
        max_rounds: int = 6,
    ) -> tuple[TurnResult, list[dict[str, Any]]]:
        data = self._request(
            "POST",
            "/engine/run_event",
            payload={
                "event": {
                    "kind": event.kind,
                    "source": event.source,
                    "prompt": event.prompt,
                    "created_at": event.created_at,
                    "metadata": dict(event.metadata),
                },
                "max_rounds": max_rounds,
                "include_trace": True,
            },
        )
        result_payload = data.get("result")
        if not isinstance(result_payload, dict):
            raise CoreAPIError("Missing result payload from core API")
        artifacts_payload = data.get("artifacts")
        artifacts: list[dict[str, Any]] = []
        if isinstance(artifacts_payload, list):
            for row in artifacts_payload:
                if isinstance(row, dict):
                    artifacts.append(row)
        return turn_result_from_dict(result_payload), artifacts

    def stream_run_event(
        self,
        event: TriggerEvent,
        *,
        max_rounds: int = 6,
        on_stream=None,
        on_artifact=None,
    ) -> TurnResult:
        payload = {
            "event": {
                "kind": event.kind,
                "source": event.source,
                "prompt": event.prompt,
                "created_at": event.created_at,
                "metadata": dict(event.metadata),
            },
            "max_rounds": max_rounds,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = urljoin(self.base_url, "engine/run_event_stream")
        request = Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return self._consume_run_event_stream(
                    response=response,
                    on_stream=on_stream,
                    on_artifact=on_artifact,
                )
        except Exception as error:  # noqa: BLE001
            raise CoreAPIError(f"Core API request failed: {error}") from error

    def reset_session(self) -> None:
        self._request("POST", "/engine/session/reset", payload={})

    def add_prompt(self, prompt: str) -> list[str]:
        data = self._request("POST", "/history/add", payload={"prompt": prompt})
        prompts = data.get("prompts")
        if not isinstance(prompts, list):
            return []
        return [str(item) for item in prompts]

    def list_prompts(self, *, limit: int | None = None) -> list[str]:
        query: dict[str, str] = {}
        if limit is not None:
            query["limit"] = str(limit)
        data = self._request("GET", "/history/list", query=query)
        prompts = data.get("prompts")
        if not isinstance(prompts, list):
            return []
        return [str(item) for item in prompts]

    def list_profile_names(self) -> list[str]:
        data = self._request("GET", "/agent/profiles")
        names = data.get("names")
        if not isinstance(names, list):
            return []
        return [str(name) for name in names]

    def get_active_profile(self, *, repo_root: Path) -> AgentProfile:
        data = self._request("GET", "/agent/active")
        profile = data.get("profile")
        if not isinstance(profile, dict):
            raise CoreAPIError("Missing active profile in core API response")
        return profile_from_dict(profile, repo_root=repo_root)

    def set_active_profile(self, name: str, *, repo_root: Path) -> AgentProfile:
        data = self._request("POST", "/agent/activate", payload={"name": name})
        profile = data.get("profile")
        if not isinstance(profile, dict):
            raise CoreAPIError("Missing profile in activate response")
        return profile_from_dict(profile, repo_root=repo_root)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        relative = path.lstrip("/")
        url = urljoin(self.base_url, relative)
        if query:
            url = f"{url}?{urlencode(query)}"

        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(url=url, data=body, method=method.upper(), headers=headers)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except Exception as error:  # noqa: BLE001
            raise CoreAPIError(f"Core API request failed: {error}") from error

        try:
            payload_out = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CoreAPIError(f"Core API returned invalid JSON: {error}") from error

        if not isinstance(payload_out, dict):
            raise CoreAPIError("Core API returned a non-object payload")
        if not payload_out.get("ok", False):
            detail = str(payload_out.get("error", "unknown error"))
            raise CoreAPIError(detail)

        data = payload_out.get("data")
        if not isinstance(data, dict):
            raise CoreAPIError("Core API response is missing 'data' object")
        return data

    @staticmethod
    def _consume_run_event_stream(*, response, on_stream=None, on_artifact=None) -> TurnResult:
        event_name = "message"
        data_lines: list[str] = []
        pending_result: TurnResult | None = None

        def _dispatch() -> None:
            nonlocal event_name, data_lines, pending_result
            if not data_lines:
                event_name = "message"
                return
            raw_payload = "\n".join(data_lines).strip()
            data_lines = []
            if not raw_payload:
                event_name = "message"
                return
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError as error:
                raise CoreAPIError(f"Invalid SSE payload from core API: {error}") from error
            if not isinstance(payload, dict):
                raise CoreAPIError("Invalid SSE payload from core API: expected object")

            if event_name == "stream":
                chunk = payload.get("chunk")
                if on_stream is not None and chunk is not None:
                    on_stream(str(chunk))
            elif event_name == "artifact":
                if on_artifact is not None:
                    on_artifact(payload)
            elif event_name == "result":
                result_payload = payload.get("result")
                if not isinstance(result_payload, dict):
                    raise CoreAPIError("Missing result payload in stream response")
                pending_result = turn_result_from_dict(result_payload)
            elif event_name == "error":
                raise CoreAPIError(str(payload.get("message", "unknown stream error")))
            event_name = "message"

        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            if line.endswith("\r"):
                line = line[:-1]
            if not line:
                _dispatch()
                if pending_result is not None:
                    return pending_result
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip() or "message"
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue

        if data_lines:
            _dispatch()
        if pending_result is None:
            raise CoreAPIError("Stream ended without result payload")
        return pending_result


class RemotePromptHistoryStore:
    """History proxy that delegates to core API."""

    def __init__(self, client: CoreAPIClient) -> None:
        self.client = client
        self.enabled = True

    def list_prompts(self, *, limit: int | None = None) -> list[str]:
        return self.client.list_prompts(limit=limit)

    def add_prompt(self, prompt: str) -> list[str]:
        return self.client.add_prompt(prompt)


class RemoteTriggerGateway:
    """Trigger gateway proxy that delegates event collection to core API."""

    def __init__(self, client: CoreAPIClient) -> None:
        self.client = client

    def collect(self, *, trigger: str, prompt: str | None = None, consume: bool = True) -> list[TriggerEvent]:
        return self.client.collect_events(trigger=trigger, prompt=prompt, consume=consume)


class _NoOpLLMController:
    def stop_stream(self) -> None:
        return None


class RemoteAgentEngine:
    """Engine proxy that executes events in remote core API."""

    def __init__(self, client: CoreAPIClient) -> None:
        self.client = client
        self.llm = _NoOpLLMController()

    def run_event(
        self,
        event: TriggerEvent,
        *,
        max_rounds: int = 6,
        on_stream=None,
        on_artifact=None,
    ) -> TurnResult:
        if on_stream is None and on_artifact is None:
            return self.client.run_event(event, max_rounds=max_rounds)
        return self.client.stream_run_event(
            event,
            max_rounds=max_rounds,
            on_stream=on_stream,
            on_artifact=on_artifact,
        )

    def reset_session(self) -> None:
        self.client.reset_session()


class RemoteAgentProfileManager:
    """Agent profile manager proxy for remote core API."""

    is_remote = True

    def __init__(self, client: CoreAPIClient, *, repo_root: Path) -> None:
        self.client = client
        self.repo_root = repo_root

    def list_profile_names(self) -> list[str]:
        return self.client.list_profile_names()

    def get_active_profile(self) -> AgentProfile:
        return self.client.get_active_profile(repo_root=self.repo_root)

    def set_active_profile(self, name: str) -> AgentProfile:
        return self.client.set_active_profile(name, repo_root=self.repo_root)
