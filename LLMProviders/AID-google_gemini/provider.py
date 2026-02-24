"""Google Gemini provider adapter with a uniform API surface."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from typing import Any

from agent.provider_base import ProviderError

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.IGNORECASE)


class GeminiProviderError(ProviderError):
    """Raised when the Gemini provider cannot complete a request."""


class GeminiProvider:
    """Google Gemini provider with non-streaming and streaming interfaces."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
    ) -> None:
        resolved_api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not resolved_api_key:
            raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY or pass api_key.")
        self.api_key = resolved_api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_models(self, *, page_size: int = 100) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        next_token: str | None = None
        while True:
            query: dict[str, Any] = {"pageSize": page_size}
            if next_token:
                query["pageToken"] = next_token
            payload = self._request_json("GET", "/models", query=query)
            models.extend(payload.get("models", []))
            next_token = payload.get("nextPageToken")
            if not next_token:
                break
        return models

    def list_model_names(self, *, page_size: int = 100) -> list[str]:
        names: list[str] = []
        for model in self.list_models(page_size=page_size):
            name = model.get("name")
            if isinstance(name, str):
                names.append(name)
        return names

    def generate_content(
        self,
        model: str,
        contents: Any,
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
    ) -> dict[str, Any]:
        payload = self._build_generate_payload(
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
            cached_content=cached_content,
        )
        endpoint = f"/{self._normalize_model_name(model)}:generateContent"
        return self._request_json("POST", endpoint, payload=payload)

    def stream_generate_content(
        self,
        model: str,
        contents: Any,
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        payload = self._build_generate_payload(
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
            cached_content=cached_content,
        )
        endpoint = f"/{self._normalize_model_name(model)}:streamGenerateContent"
        url = self._build_url(endpoint, query={"alt": "sse"})

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_payload in _iter_sse_payloads(response):
                    if raw_payload == "[DONE]":
                        break
                    try:
                        yield json.loads(raw_payload)
                    except json.JSONDecodeError as error:
                        raise GeminiProviderError(
                            f"Invalid JSON payload in Gemini stream: {raw_payload[:300]}"
                        ) from error
        except urllib.error.HTTPError as error:
            raise GeminiProviderError(self._format_http_error(error)) from error
        except urllib.error.URLError as error:
            raise GeminiProviderError(f"Gemini stream connection error: {error.reason}") from error

    def generate_text(
        self,
        model: str,
        prompt: str,
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
    ) -> str:
        response = self.generate_content(
            model=model,
            contents=prompt,
            system_instruction=system_instruction,
            generation_config=generation_config,
        )
        return extract_text_from_response(response)

    def stream_generate_text(
        self,
        model: str,
        prompt: str,
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        for event in self.stream_generate_content(
            model=model,
            contents=prompt,
            system_instruction=system_instruction,
            generation_config=generation_config,
        ):
            for text_chunk in _extract_text_parts(event):
                yield text_chunk

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        inferred_system_instruction, gemini_contents = _split_system_and_contents(messages)
        merged_system_instruction = _merge_system_instruction(
            inferred_system_instruction, system_instruction
        )
        return self.generate_content(
            model=model,
            contents=gemini_contents,
            system_instruction=merged_system_instruction,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
        )

    def chat_text(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
    ) -> str:
        response = self.chat(
            model=model,
            messages=messages,
            system_instruction=system_instruction,
            generation_config=generation_config,
        )
        return extract_text_from_response(response)

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        inferred_system_instruction, gemini_contents = _split_system_and_contents(messages)
        merged_system_instruction = _merge_system_instruction(
            inferred_system_instruction, system_instruction
        )
        return self.stream_generate_content(
            model=model,
            contents=gemini_contents,
            system_instruction=merged_system_instruction,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
        )

    def stream_chat_text(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        for event in self.stream_chat(
            model=model,
            messages=messages,
            system_instruction=system_instruction,
            generation_config=generation_config,
        ):
            for text_chunk in _extract_text_parts(event):
                yield text_chunk

    def count_tokens(
        self,
        model: str,
        contents: Any,
        *,
        system_instruction: Any = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"contents": _normalize_contents(contents)}
        normalized_system = _normalize_system_instruction(system_instruction)
        if normalized_system:
            payload["systemInstruction"] = normalized_system
        if tools:
            payload["tools"] = tools
        endpoint = f"/{self._normalize_model_name(model)}:countTokens"
        return self._request_json("POST", endpoint, payload=payload)

    def embed_content(
        self,
        model: str,
        content: Any,
        *,
        task_type: str | None = None,
        title: str | None = None,
        output_dimensionality: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"content": _normalize_embedding_content(content)}
        if task_type:
            payload["taskType"] = task_type
        if title:
            payload["title"] = title
        if output_dimensionality is not None:
            payload["outputDimensionality"] = output_dimensionality
        endpoint = f"/{self._normalize_model_name(model)}:embedContent"
        return self._request_json("POST", endpoint, payload=payload)

    def batch_embed_contents(
        self,
        model: str,
        contents: list[Any],
        *,
        task_type: str | None = None,
        output_dimensionality: int | None = None,
    ) -> dict[str, Any]:
        normalized_model = self._normalize_model_name(model)
        requests: list[dict[str, Any]] = []
        for content in contents:
            item: dict[str, Any] = {
                "model": normalized_model,
                "content": _normalize_embedding_content(content),
            }
            if task_type:
                item["taskType"] = task_type
            if output_dimensionality is not None:
                item["outputDimensionality"] = output_dimensionality
            requests.append(item)
        payload = {"requests": requests}
        endpoint = f"/{normalized_model}:batchEmbedContents"
        return self._request_json("POST", endpoint, payload=payload)

    def _build_generate_payload(
        self,
        *,
        contents: Any,
        system_instruction: Any = None,
        generation_config: dict[str, Any] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"contents": _normalize_contents(contents)}
        normalized_system = _normalize_system_instruction(system_instruction)
        if normalized_system:
            payload["systemInstruction"] = normalized_system
        if generation_config:
            payload["generationConfig"] = generation_config
        if safety_settings:
            payload["safetySettings"] = safety_settings
        if tools:
            payload["tools"] = tools
        if tool_config:
            payload["toolConfig"] = tool_config
        if cached_content:
            payload["cachedContent"] = cached_content
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
            headers={"Content-Type": "application/json"},
            method=method,
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            raise GeminiProviderError(self._format_http_error(error)) from error
        except urllib.error.URLError as error:
            raise GeminiProviderError(f"Gemini connection error: {error.reason}") from error

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as error:
            raise GeminiProviderError(f"Gemini returned invalid JSON: {body[:300]}") from error

        if isinstance(parsed, dict):
            return parsed
        raise GeminiProviderError("Gemini returned a non-object JSON response.")

    def _build_url(self, path: str, *, query: dict[str, Any] | None = None) -> str:
        query_params: dict[str, Any] = {"key": self.api_key}
        if query:
            for key, value in query.items():
                if value is not None:
                    query_params[key] = value
        encoded_query = urllib.parse.urlencode(query_params, doseq=True)
        return f"{self.base_url}{path}?{encoded_query}"

    def _normalize_model_name(self, model: str) -> str:
        stripped = model.strip()
        if not stripped:
            raise ValueError("Model name cannot be empty.")
        if stripped.startswith("models/"):
            return stripped
        return f"models/{stripped}"

    @staticmethod
    def _format_http_error(error: urllib.error.HTTPError) -> str:
        body = error.read().decode("utf-8", errors="replace")
        message = body
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                maybe_error = parsed.get("error")
                if isinstance(maybe_error, dict):
                    maybe_message = maybe_error.get("message")
                    if isinstance(maybe_message, str):
                        message = maybe_message
        except json.JSONDecodeError:
            pass
        return f"Gemini API HTTP {error.code}: {message}"


def extract_text_from_response(response: dict[str, Any]) -> str:
    text = "".join(_extract_text_parts(response)).strip()
    if text:
        return text
    raise GeminiProviderError("No text content found in Gemini response.")


def _extract_text_parts(response: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    candidates = response.get("candidates", [])
    if not isinstance(candidates, list):
        return chunks
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            continue
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return chunks


def _split_system_and_contents(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    system_parts: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", "user")).lower().strip()
        content = message.get("content", "")
        parts = _normalize_parts(content)
        if role == "system":
            system_parts.extend(parts)
            continue

        gemini_role = "model" if role in {"assistant", "model"} else "user"
        contents.append({"role": gemini_role, "parts": parts})

    if not contents:
        raise ValueError("At least one non-system message is required for chat.")

    system_instruction = {"parts": system_parts} if system_parts else None
    return system_instruction, contents


def _merge_system_instruction(
    inferred: dict[str, Any] | None,
    explicit: Any,
) -> dict[str, Any] | None:
    normalized_explicit = _normalize_system_instruction(explicit)
    if inferred and normalized_explicit:
        return {"parts": inferred.get("parts", []) + normalized_explicit.get("parts", [])}
    return normalized_explicit or inferred


def _normalize_contents(contents: Any) -> list[dict[str, Any]]:
    if isinstance(contents, str):
        return [{"role": "user", "parts": [{"text": contents}]}]

    if isinstance(contents, dict):
        if "parts" in contents:
            return [_normalize_content_item(contents)]
        if "role" in contents and "content" in contents:
            role = str(contents["role"]).lower().strip()
            mapped_role = "model" if role in {"assistant", "model"} else "user"
            return [{"role": mapped_role, "parts": _normalize_parts(contents["content"])}]
        raise ValueError("Content dict must contain either 'parts' or ('role' and 'content').")

    if not isinstance(contents, list):
        raise TypeError("contents must be str, dict, or list of dict items.")

    normalized: list[dict[str, Any]] = []
    for item in contents:
        if not isinstance(item, dict):
            raise TypeError("Each contents item must be a dict.")
        if "parts" in item:
            normalized.append(_normalize_content_item(item))
            continue
        if "role" in item and "content" in item:
            role = str(item["role"]).lower().strip()
            if role == "system":
                continue
            mapped_role = "model" if role in {"assistant", "model"} else "user"
            normalized.append({"role": mapped_role, "parts": _normalize_parts(item["content"])})
            continue
        raise ValueError("Each contents item must define 'parts' or ('role' and 'content').")

    if not normalized:
        raise ValueError("At least one content item is required.")
    return normalized


def _normalize_content_item(item: dict[str, Any]) -> dict[str, Any]:
    role = str(item.get("role", "user")).lower().strip()
    normalized_role = "model" if role in {"assistant", "model"} else "user"
    return {"role": normalized_role, "parts": _normalize_parts(item.get("parts", []))}


def _normalize_system_instruction(system_instruction: Any) -> dict[str, Any] | None:
    if system_instruction is None:
        return None
    return {"parts": _normalize_parts(system_instruction)}


def _normalize_embedding_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        if "parts" in content:
            return {"parts": _normalize_parts(content["parts"])}
        if "content" in content and isinstance(content["content"], str):
            return {"parts": [{"text": content["content"]}]}
    return {"parts": _normalize_parts(content)}


def _normalize_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]

    if isinstance(content, dict):
        if "parts" in content and isinstance(content["parts"], list):
            return [_normalize_part(part) for part in content["parts"]]
        return [_normalize_part(content)]

    if not isinstance(content, list):
        raise TypeError("Message content must be str, dict, or list.")

    normalized = [_normalize_part(part) for part in content]
    if not normalized:
        raise ValueError("Message parts cannot be empty.")
    return normalized


def _normalize_part(part: Any) -> dict[str, Any]:
    if not isinstance(part, dict):
        raise TypeError("Each part must be a dict.")

    for supported_key in (
        "text",
        "inline_data",
        "file_data",
        "function_call",
        "function_response",
        "executable_code",
        "code_execution_result",
    ):
        if supported_key in part:
            return part

    part_type = part.get("type")

    if part_type in {"text", "input_text"}:
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Text part requires a string in 'text'.")
        return {"text": text}

    if part_type in {"image_url", "input_image"}:
        url: str | None = None
        mime_type = part.get("mime_type")

        image_url_value = part.get("image_url")
        if isinstance(image_url_value, dict):
            raw_url = image_url_value.get("url")
            if isinstance(raw_url, str):
                url = raw_url
        elif isinstance(image_url_value, str):
            url = image_url_value
        elif isinstance(part.get("url"), str):
            url = part["url"]

        if isinstance(part.get("image_base64"), str):
            base64_data = part["image_base64"]
            resolved_mime = mime_type if isinstance(mime_type, str) else "image/png"
            return {"inline_data": {"mime_type": resolved_mime, "data": base64_data}}

        if not isinstance(url, str):
            raise ValueError("Image part requires image_url.url, image_url, url, or image_base64.")
        return _image_url_to_part(url, mime_type if isinstance(mime_type, str) else None)

    raise ValueError(f"Unsupported part format: {part}")


def _image_url_to_part(url: str, mime_type: str | None = None) -> dict[str, Any]:
    match = DATA_URL_RE.match(url)
    if match:
        return {
            "inline_data": {
                "mime_type": match.group("mime"),
                "data": match.group("data"),
            }
        }

    resolved_mime = mime_type or "application/octet-stream"
    return {"file_data": {"mime_type": resolved_mime, "file_uri": url}}


def _iter_sse_payloads(stream: Iterable[bytes]) -> Iterator[str]:
    event_lines: list[str] = []

    for raw_line in stream:
        line = raw_line.decode("utf-8").rstrip("\n")
        if line.endswith("\r"):
            line = line[:-1]

        if not line:
            if event_lines:
                payload = "\n".join(
                    entry[5:].lstrip() for entry in event_lines if entry.startswith("data:")
                ).strip()
                if payload:
                    yield payload
                event_lines = []
            continue

        if line.startswith(":"):
            continue
        event_lines.append(line)

    if event_lines:
        payload = "\n".join(
            entry[5:].lstrip() for entry in event_lines if entry.startswith("data:")
        ).strip()
        if payload:
            yield payload
