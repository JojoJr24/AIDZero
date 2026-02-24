"""Shared adapter for OpenAI-compatible chat/embedding providers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from typing import Any

from agent.provider_base import ProviderError


class OpenAICompatibleProviderError(ProviderError):
    """Raised when an OpenAI-compatible provider cannot complete a request."""


class OpenAICompatibleProvider:
    """Reusable provider adapter for APIs compatible with OpenAI endpoints."""

    def __init__(
        self,
        *,
        provider_label: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        require_api_key: bool = False,
        timeout: int = 120,
        error_cls: type[OpenAICompatibleProviderError] = OpenAICompatibleProviderError,
    ) -> None:
        resolved_base_url = base_url.strip().rstrip("/")
        if not resolved_base_url:
            raise ValueError("base_url cannot be empty.")

        resolved_api_key = (api_key or "").strip()
        if not resolved_api_key and api_key_env:
            resolved_api_key = os.getenv(api_key_env, "").strip()
        if require_api_key and not resolved_api_key:
            if api_key_env:
                raise ValueError(f"Missing API key. Set {api_key_env} or pass api_key.")
            raise ValueError("Missing API key. Pass api_key.")

        self.provider_label = provider_label
        self.base_url = resolved_base_url
        self.api_key = resolved_api_key
        self.timeout = timeout
        self.error_cls = error_cls

    def list_models(self, *, page_size: int = 100) -> list[dict[str, Any]]:
        del page_size  # Not supported consistently by OpenAI-compatible servers.
        payload = self._request_json("GET", "/models")
        model_items = payload.get("data")
        if not isinstance(model_items, list):
            raise self.error_cls(f"{self.provider_label} returned an invalid models payload.")
        return [item for item in model_items if isinstance(item, dict)]

    def list_model_names(self, *, page_size: int = 100) -> list[str]:
        model_names: list[str] = []
        for item in self.list_models(page_size=page_size):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id.strip():
                model_names.append(model_id)
        return model_names

    def generate_content(
        self,
        model: str,
        contents: Any,
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = self._build_chat_payload(
            model=model,
            messages=_normalize_messages(contents, system_instruction=system_instruction),
            generation_config=generation_config,
            tools=tools,
            tool_choice=tool_choice,
            extra=kwargs,
        )
        return self._request_json("POST", "/chat/completions", payload=payload)

    def stream_generate_content(
        self,
        model: str,
        contents: Any,
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        payload = self._build_chat_payload(
            model=model,
            messages=_normalize_messages(contents, system_instruction=system_instruction),
            generation_config=generation_config,
            tools=tools,
            tool_choice=tool_choice,
            extra={**kwargs, "stream": True},
        )
        return self._stream_json_events("POST", "/chat/completions", payload=payload)

    def generate_text(
        self,
        model: str,
        prompt: str,
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        response = self.generate_content(
            model=model,
            contents=prompt,
            system_instruction=system_instruction,
            generation_config=generation_config,
            **kwargs,
        )
        return extract_text_from_chat_completion(response, error_cls=self.error_cls)

    def stream_generate_text(
        self,
        model: str,
        prompt: str,
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        for event in self.stream_generate_content(
            model=model,
            contents=prompt,
            system_instruction=system_instruction,
            generation_config=generation_config,
            **kwargs,
        ):
            for text_chunk in _extract_stream_text(event):
                yield text_chunk

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = self._build_chat_payload(
            model=model,
            messages=_normalize_messages(messages, system_instruction=None),
            generation_config=generation_config,
            tools=tools,
            tool_choice=tool_choice,
            extra=kwargs,
        )
        return self._request_json("POST", "/chat/completions", payload=payload)

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        payload = self._build_chat_payload(
            model=model,
            messages=_normalize_messages(messages, system_instruction=None),
            generation_config=generation_config,
            tools=tools,
            tool_choice=tool_choice,
            extra={**kwargs, "stream": True},
        )
        return self._stream_json_events("POST", "/chat/completions", payload=payload)

    def count_tokens(self, model: str, contents: Any, **kwargs: Any) -> dict[str, Any]:
        del model, contents, kwargs
        raise self.error_cls(
            f"{self.provider_label} does not expose a standard token counting endpoint "
            "in the OpenAI-compatible API."
        )

    def embed_content(
        self,
        model: str,
        content: Any,
        *,
        dimensions: int | None = None,
        encoding_format: str | None = None,
        user: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": _normalize_model_name(model),
            "input": _normalize_embedding_input(content),
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if encoding_format:
            payload["encoding_format"] = encoding_format
        if user:
            payload["user"] = user
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._request_json("POST", "/embeddings", payload=payload)

    def _build_chat_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        generation_config: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": _normalize_model_name(model),
            "messages": messages,
        }
        if generation_config:
            for key, value in generation_config.items():
                if value is not None:
                    payload[key] = value
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        return payload

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url=self._build_url(path, query=query),
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers=self._build_headers(),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            raise self.error_cls(self._format_http_error(error)) from error
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", str(error))
            raise self.error_cls(f"{self.provider_label} connection error: {reason}") from error

        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError as error:
            raise self.error_cls(
                f"{self.provider_label} returned invalid JSON: {raw_body[:300]}"
            ) from error

        if isinstance(parsed, dict):
            return parsed
        raise self.error_cls(f"{self.provider_label} returned a non-object JSON response.")

    def _stream_json_events(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        headers = self._build_headers()
        headers["Accept"] = "text/event-stream"
        request = urllib.request.Request(
            url=self._build_url(path),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_payload in _iter_sse_payloads(response):
                    if raw_payload == "[DONE]":
                        break
                    try:
                        parsed = json.loads(raw_payload)
                    except json.JSONDecodeError as error:
                        raise self.error_cls(
                            f"{self.provider_label} stream returned invalid JSON: "
                            f"{raw_payload[:300]}"
                        ) from error
                    if isinstance(parsed, dict):
                        yield parsed
        except urllib.error.HTTPError as error:
            raise self.error_cls(self._format_http_error(error)) from error
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", str(error))
            raise self.error_cls(f"{self.provider_label} stream connection error: {reason}") from error

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_url(self, path: str, *, query: dict[str, Any] | None = None) -> str:
        normalized_path = "/" + path.lstrip("/")
        if query:
            encoded = urllib.parse.urlencode(
                {key: value for key, value in query.items() if value is not None},
                doseq=True,
            )
            if encoded:
                return f"{self.base_url}{normalized_path}?{encoded}"
        return f"{self.base_url}{normalized_path}"

    def _format_http_error(self, error: urllib.error.HTTPError) -> str:
        body = error.read().decode("utf-8", errors="replace")
        message = body
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            maybe_error = parsed.get("error")
            if isinstance(maybe_error, dict):
                maybe_message = maybe_error.get("message")
                if isinstance(maybe_message, str) and maybe_message.strip():
                    message = maybe_message
                elif isinstance(maybe_error.get("type"), str):
                    message = str(maybe_error["type"])
            elif isinstance(maybe_error, str) and maybe_error.strip():
                message = maybe_error
            elif isinstance(parsed.get("message"), str):
                message = str(parsed["message"])
        return f"{self.provider_label} API HTTP {error.code}: {message}"


def extract_text_from_chat_completion(
    response: dict[str, Any],
    *,
    error_cls: type[OpenAICompatibleProviderError] = OpenAICompatibleProviderError,
) -> str:
    chunks: list[str] = []
    choices = response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                chunks.extend(_extract_content_text(message.get("content")))
            text_value = choice.get("text")
            if isinstance(text_value, str):
                chunks.append(text_value)
    text = "".join(chunks).strip()
    if text:
        return text
    raise error_cls("No text content found in OpenAI-compatible response.")


def _extract_stream_text(payload: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return chunks
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict):
            chunks.extend(_extract_content_text(delta.get("content")))
        text_value = choice.get("text")
        if isinstance(text_value, str):
            chunks.append(text_value)
    return chunks


def _normalize_model_name(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        raise ValueError("Model name cannot be empty.")
    return normalized


def _normalize_messages(contents: Any, *, system_instruction: Any = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]]
    if isinstance(contents, str):
        messages = [{"role": "user", "content": contents}]
    elif isinstance(contents, dict):
        if isinstance(contents.get("messages"), list):
            messages = _normalize_message_list(contents["messages"])
        elif "role" in contents and "content" in contents:
            messages = [_normalize_message(contents)]
        else:
            messages = [{"role": "user", "content": _safe_json_text(contents)}]
    elif isinstance(contents, list):
        if contents and all(isinstance(item, dict) and "role" in item for item in contents):
            messages = _normalize_message_list(contents)
        else:
            messages = [{"role": "user", "content": _normalize_content_value(contents)}]
    else:
        messages = [{"role": "user", "content": _safe_json_text(contents)}]

    if system_instruction is not None:
        messages = [{"role": "system", "content": _normalize_content_value(system_instruction)}] + messages

    if not messages:
        raise ValueError("At least one message is required.")
    return messages


def _normalize_message_list(messages: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in messages:
        if isinstance(item, dict):
            normalized.append(_normalize_message(item))
        else:
            normalized.append({"role": "user", "content": _normalize_content_value(item)})
    return normalized


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role", "user")).strip().lower() or "user"
    if role not in {"system", "user", "assistant", "tool", "developer"}:
        role = "user"

    normalized: dict[str, Any] = {"role": role}
    content = message.get("content")
    if role == "assistant" and content is None and "tool_calls" in message:
        normalized["content"] = None
    else:
        normalized["content"] = _normalize_content_value(content)

    for key in ("name", "tool_call_id", "tool_calls"):
        if key in message:
            normalized[key] = message[key]
    return normalized


def _normalize_content_value(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized_parts: list[Any] = []
        for part in content:
            if isinstance(part, dict):
                normalized_parts.append(part)
            elif isinstance(part, str):
                normalized_parts.append({"type": "text", "text": part})
            else:
                normalized_parts.append({"type": "text", "text": _safe_json_text(part)})
        return normalized_parts
    if isinstance(content, dict):
        if "type" in content and "text" in content:
            return [content]
        return _safe_json_text(content)
    return _safe_json_text(content)


def _normalize_embedding_input(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized: list[str] = []
        for item in content:
            if isinstance(item, str):
                normalized.append(item)
            else:
                normalized.append(_safe_json_text(item))
        return normalized
    return _safe_json_text(content)


def _extract_content_text(content: Any) -> list[str]:
    chunks: list[str] = []
    if isinstance(content, str):
        chunks.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text_value = part.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
                elif isinstance(part.get("content"), str):
                    chunks.append(part["content"])
            elif isinstance(part, str):
                chunks.append(part)
    elif isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            chunks.append(text_value)
    return chunks


def _safe_json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _iter_sse_payloads(stream: Iterable[bytes]) -> Iterator[str]:
    event_lines: list[str] = []
    for raw_line in stream:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
        if line.endswith("\r"):
            line = line[:-1]

        if not line:
            if event_lines:
                payload = "\n".join(
                    row[5:].lstrip() for row in event_lines if row.startswith("data:")
                ).strip()
                if payload:
                    yield payload
                event_lines.clear()
            continue

        if line.startswith(":"):
            continue
        event_lines.append(line)

    if event_lines:
        payload = "\n".join(row[5:].lstrip() for row in event_lines if row.startswith("data:")).strip()
        if payload:
            yield payload

