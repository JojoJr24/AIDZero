"""Anthropic Claude provider adapter with a uniform API surface."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from typing import Any

from LLMProviders.provider_base import (
    ProviderError,
    normalize_tool_result_content,
    parse_json_object,
)

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


class ClaudeProviderError(ProviderError):
    """Raised when the Claude provider cannot complete a request."""


class ClaudeProvider:
    """Anthropic Claude provider with non-streaming and streaming interfaces."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        anthropic_version: str | None = None,
        timeout: int = 120,
    ) -> None:
        resolved_api_key = (api_key or os.getenv("ANTHROPIC_API_KEY", "")).strip()
        if not resolved_api_key:
            raise ValueError("Missing Anthropic API key. Set ANTHROPIC_API_KEY or pass api_key.")

        resolved_base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)).strip()
        if not resolved_base_url:
            resolved_base_url = DEFAULT_BASE_URL

        resolved_version = (
            anthropic_version or os.getenv("ANTHROPIC_VERSION", DEFAULT_ANTHROPIC_VERSION)
        ).strip()
        if not resolved_version:
            resolved_version = DEFAULT_ANTHROPIC_VERSION

        self.api_key = resolved_api_key
        self.base_url = resolved_base_url.rstrip("/")
        self.anthropic_version = resolved_version
        self.timeout = timeout
        self._stream_lock = threading.Lock()
        self._active_stream_response: Any | None = None
        self._stop_requested = False

    def list_models(self, *, page_size: int = 100) -> list[dict[str, Any]]:
        payload = self._request_json("GET", "/models", query={"limit": page_size})
        models = payload.get("data")
        if not isinstance(models, list):
            raise ClaudeProviderError("Anthropic returned an invalid models payload.")
        return [item for item in models if isinstance(item, dict)]

    def list_model_names(self, *, page_size: int = 100) -> list[str]:
        model_names: list[str] = []
        for item in self.list_models(page_size=page_size):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id.strip():
                model_names.append(model_id.strip())
        return model_names

    def generate_content(
        self,
        model: str,
        contents: Any,
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        system_text, messages = _normalize_messages(contents, system_instruction=system_instruction)
        payload = _build_messages_payload(
            model=model,
            messages=messages,
            system_text=system_text,
            generation_config=generation_config,
            extra=kwargs,
        )
        return self._request_json("POST", "/messages", payload=payload)

    def stream_generate_content(
        self,
        model: str,
        contents: Any,
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        system_text, messages = _normalize_messages(contents, system_instruction=system_instruction)
        payload = _build_messages_payload(
            model=model,
            messages=messages,
            system_text=system_text,
            generation_config=generation_config,
            extra={**kwargs, "stream": True},
        )
        return self._stream_json_events("POST", "/messages", payload=payload)

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
        return extract_text_from_response(response)

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
            for chunk in _extract_stream_text(event):
                yield chunk

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        generation_config: dict[str, Any] | None = None,
        system_instruction: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        system_text, normalized_messages = _normalize_messages(
            messages,
            system_instruction=system_instruction,
        )
        payload = _build_messages_payload(
            model=model,
            messages=normalized_messages,
            system_text=system_text,
            generation_config=generation_config,
            extra=kwargs,
        )
        return self._request_json("POST", "/messages", payload=payload)

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        generation_config: dict[str, Any] | None = None,
        system_instruction: Any = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        system_text, normalized_messages = _normalize_messages(
            messages,
            system_instruction=system_instruction,
        )
        payload = _build_messages_payload(
            model=model,
            messages=normalized_messages,
            system_text=system_text,
            generation_config=generation_config,
            extra={**kwargs, "stream": True},
        )
        return self._stream_json_events("POST", "/messages", payload=payload)

    def stop_stream(self) -> None:
        with self._stream_lock:
            self._stop_requested = True
            response = self._active_stream_response
        if response is None:
            return
        close = getattr(response, "close", None)
        if callable(close):
            close()

    def count_tokens(self, model: str, contents: Any, **kwargs: Any) -> dict[str, Any]:
        system_instruction = kwargs.pop("system_instruction", None)
        system_text, messages = _normalize_messages(contents, system_instruction=system_instruction)
        payload: dict[str, Any] = {
            "model": _normalize_model_name(model),
            "messages": messages,
        }
        if system_text:
            payload["system"] = system_text
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return self._request_json("POST", "/messages/count_tokens", payload=payload)

    def embed_content(self, model: str, content: Any, **kwargs: Any) -> dict[str, Any]:
        del model, content, kwargs
        raise ClaudeProviderError("Anthropic does not expose embeddings in the Messages API.")

    def supports_tool_calling(self) -> bool:
        return True

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        content = response.get("content")
        if not isinstance(content, list):
            return tool_calls

        for index, block in enumerate(content, start=1):
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = f"tool_call_{index}"
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name.strip(),
                        "arguments": json.dumps(parse_json_object(block.get("input")), ensure_ascii=False),
                    },
                }
            )
        return tool_calls

    def build_tool_result_message(
        self,
        *,
        tool_call_id: str,
        name: str,
        result: Any,
    ) -> dict[str, Any]:
        normalized_id = tool_call_id.strip()
        if not normalized_id:
            raise ValueError("tool_call_id cannot be empty.")
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("name cannot be empty.")
        return {
            "role": "tool",
            "tool_call_id": normalized_id,
            "name": normalized_name,
            "content": normalize_tool_result_content(result),
        }

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
            raise ClaudeProviderError(self._format_http_error(error)) from error
        except urllib.error.URLError as error:
            raise ClaudeProviderError(f"Anthropic connection error: {error.reason}") from error

        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError as error:
            raise ClaudeProviderError(f"Anthropic returned invalid JSON: {raw_body[:300]}") from error

        if not isinstance(parsed, dict):
            raise ClaudeProviderError("Anthropic returned a non-object JSON response.")
        return parsed

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
                self._register_active_stream(response)
                try:
                    for raw_payload in _iter_sse_payloads(response):
                        if self._is_stop_requested():
                            break
                        if raw_payload == "[DONE]":
                            break
                        try:
                            parsed = json.loads(raw_payload)
                        except json.JSONDecodeError as error:
                            raise ClaudeProviderError(
                                f"Anthropic stream returned invalid JSON: {raw_payload[:300]}"
                            ) from error
                        if isinstance(parsed, dict):
                            yield parsed
                finally:
                    self._clear_active_stream(response)
        except urllib.error.HTTPError as error:
            if self._consume_stop_requested():
                return
            raise ClaudeProviderError(self._format_http_error(error)) from error
        except urllib.error.URLError as error:
            if self._consume_stop_requested():
                return
            raise ClaudeProviderError(f"Anthropic stream connection error: {error.reason}") from error
        except Exception:
            if self._consume_stop_requested():
                return
            raise
        self._consume_stop_requested()

    def _build_url(self, path: str, *, query: dict[str, Any] | None = None) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        url = f"{self.base_url}{normalized_path}"
        if not query:
            return url

        cleaned_query = {key: value for key, value in query.items() if value is not None}
        if not cleaned_query:
            return url
        return f"{url}?{urllib.parse.urlencode(cleaned_query)}"

    def _build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
        }

    @staticmethod
    def _format_http_error(error: urllib.error.HTTPError) -> str:
        try:
            body = error.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            body = ""
        if body:
            return f"Anthropic HTTP {error.code}: {body}"
        return f"Anthropic HTTP {error.code}: {error.reason}"

    def _register_active_stream(self, response: Any) -> None:
        with self._stream_lock:
            self._active_stream_response = response
            self._stop_requested = False

    def _clear_active_stream(self, response: Any) -> None:
        with self._stream_lock:
            if self._active_stream_response is response:
                self._active_stream_response = None

    def _is_stop_requested(self) -> bool:
        with self._stream_lock:
            return self._stop_requested

    def _consume_stop_requested(self) -> bool:
        with self._stream_lock:
            requested = self._stop_requested
            self._stop_requested = False
            return requested


def extract_text_from_response(payload: dict[str, Any]) -> str:
    content_blocks = payload.get("content")
    if not isinstance(content_blocks, list):
        raise ClaudeProviderError("Anthropic response missing 'content' list.")

    chunks: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            chunks.append(text)

    text = "".join(chunks).strip()
    if text:
        return text
    raise ClaudeProviderError("Anthropic response did not include text content.")


def _build_messages_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_text: str | None,
    generation_config: dict[str, Any] | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": _normalize_model_name(model),
        "messages": messages,
        "max_tokens": 1024,
    }
    if system_text:
        payload["system"] = system_text

    if generation_config:
        for key in ("max_tokens", "temperature", "top_k", "top_p", "stop_sequences", "metadata"):
            if key in generation_config and generation_config[key] is not None:
                payload[key] = generation_config[key]

    if extra:
        for key, value in extra.items():
            if value is not None:
                payload[key] = value

    return payload


def _normalize_messages(
    contents: Any,
    *,
    system_instruction: Any = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    system_text = _to_text(system_instruction)
    normalized_messages: list[dict[str, Any]] = []

    if isinstance(contents, str):
        text = contents.strip()
        if text:
            normalized_messages.append({"role": "user", "content": text})
    elif isinstance(contents, dict):
        normalized_messages.append(contents)
    elif isinstance(contents, Iterable):
        for item in contents:
            if isinstance(item, dict):
                normalized_messages.append(item)
    else:
        text = _to_text(contents)
        if text:
            normalized_messages.append({"role": "user", "content": text})

    if not normalized_messages:
        raise ClaudeProviderError("Messages cannot be empty.")

    cleaned_messages: list[dict[str, Any]] = []
    for message in normalized_messages:
        role = str(message.get("role", "user")).strip().lower() or "user"
        if role == "system":
            candidate_text = _to_text(message.get("content"))
            if candidate_text:
                system_text = _merge_text(system_text, candidate_text)
            continue

        if role == "assistant":
            content = _normalize_assistant_content(message)
            if content is None:
                continue
            cleaned_messages.append({"role": "assistant", "content": content})
            continue

        if role == "tool":
            content = _normalize_tool_result_message_content(message)
            if content is None:
                continue
            cleaned_messages.append({"role": "user", "content": content})
            continue

        content = _normalize_user_content(message.get("content"))
        if content is None:
            continue
        cleaned_messages.append({"role": "user", "content": content})

    if not cleaned_messages:
        raise ClaudeProviderError("Messages cannot be empty after normalization.")
    return system_text, cleaned_messages


def _normalize_user_content(content: Any) -> str | list[dict[str, Any]] | None:
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if not isinstance(tool_use_id, str) or not tool_use_id.strip():
                    continue
                block_content = block.get("content")
                if isinstance(block_content, str):
                    payload_content: Any = block_content
                else:
                    payload_content = normalize_tool_result_content(block_content)
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id.strip(),
                        "content": payload_content,
                    }
                )
                continue
            if block_type != "text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                blocks.append({"type": "text", "text": text})
        if blocks:
            return blocks
        return None

    text = _to_text(content)
    return text if text else None


def _normalize_assistant_content(message: dict[str, Any]) -> str | list[dict[str, Any]] | None:
    content = message.get("content")
    blocks: list[dict[str, Any]] = []

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                name = block.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                tool_id = block.get("id")
                if not isinstance(tool_id, str) or not tool_id.strip():
                    continue
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_id.strip(),
                        "name": name.strip(),
                        "input": parse_json_object(block.get("input")),
                    }
                )
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                blocks.append({"type": "text", "text": text})
    else:
        text = _to_text(content)
        if text:
            blocks.append({"type": "text", "text": text})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for index, raw_call in enumerate(tool_calls, start=1):
            if not isinstance(raw_call, dict):
                continue
            function_payload = raw_call.get("function")
            if not isinstance(function_payload, dict):
                continue
            name = function_payload.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_id = raw_call.get("id")
            if not isinstance(tool_id, str) or not tool_id.strip():
                tool_id = f"tool_call_{index}"
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name.strip(),
                    "input": parse_json_object(function_payload.get("arguments")),
                }
            )

    if not blocks:
        return None
    return blocks


def _normalize_tool_result_message_content(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    tool_use_id = message.get("tool_call_id")
    if not isinstance(tool_use_id, str) or not tool_use_id.strip():
        return None
    content_text = normalize_tool_result_content(message.get("content"))
    return [
        {
            "type": "tool_result",
            "tool_use_id": tool_use_id.strip(),
            "content": content_text,
        }
    ]


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                chunks.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        merged = "\n".join(chunks).strip()
        return merged if merged else None
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return str(value).strip() or None


def _merge_text(primary: str | None, secondary: str | None) -> str | None:
    if primary and secondary:
        return f"{primary}\n\n{secondary}"
    return primary or secondary


def _normalize_model_name(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        raise ClaudeProviderError("model cannot be empty.")
    return normalized


def _extract_stream_text(event: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    event_type = event.get("type")
    if event_type == "content_block_delta":
        delta = event.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "text_delta":
            text = delta.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
    elif event_type == "message_delta":
        delta = event.get("delta")
        if isinstance(delta, dict):
            text = delta.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
    return chunks


def _iter_sse_payloads(response: Any) -> Iterator[str]:
    buffer: list[str] = []
    for raw_line in response:
        decoded = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not decoded:
            if buffer:
                merged = "\n".join(buffer).strip()
                if merged:
                    yield merged
                buffer = []
            continue
        if decoded.startswith("data:"):
            buffer.append(decoded[5:].strip())
    if buffer:
        merged = "\n".join(buffer).strip()
        if merged:
            yield merged


__all__ = ["ClaudeProvider", "ClaudeProviderError", "extract_text_from_response"]
