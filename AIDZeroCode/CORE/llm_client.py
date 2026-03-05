"""Provider-agnostic LLM client wrapper."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Iterator

from CORE.repo_layout import resolve_code_root


class LLMClient:
    """Loads providers from `LLMProviders/<name>/provider.py` and normalizes text output."""

    def __init__(self, *, repo_root: Path, provider_name: str, model: str) -> None:
        self.repo_root = repo_root.resolve()
        self.provider_name = provider_name.strip()
        self.model = model.strip()
        if not self.provider_name:
            raise ValueError("provider_name cannot be empty.")
        if not self.model:
            raise ValueError("model cannot be empty.")
        self.provider = self._load_provider(self.provider_name)

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        if hasattr(self.provider, "chat"):
            payload = self.provider.chat(self.model, messages, **kwargs)
            return self._extract_text(payload)
        if hasattr(self.provider, "generate_text"):
            prompt = self._flatten_messages(messages)
            text = self.provider.generate_text(self.model, prompt, **kwargs)
            return str(text).strip()
        raise RuntimeError(
            f"Provider '{self.provider_name}' does not expose chat(...) or generate_text(...)."
        )

    def complete_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        if hasattr(self.provider, "stream_chat"):
            structured_tool_calls: dict[int, dict[str, str]] = {}
            emitted_text = False
            for payload in self.provider.stream_chat(self.model, messages, **kwargs):
                for chunk in self._extract_openai_stream(payload):
                    if chunk:
                        emitted_text = True
                        yield chunk
                self._collect_openai_stream_tool_calls(payload, structured_tool_calls)
            tool_block = self._serialize_first_tool_call(structured_tool_calls)
            if tool_block and not emitted_text:
                yield tool_block
            return

        if hasattr(self.provider, "stream_generate_text"):
            prompt = self._flatten_messages(messages)
            for chunk in self.provider.stream_generate_text(self.model, prompt, **kwargs):
                text = str(chunk)
                if text:
                    yield text
            return

        # Fallback to non-streaming providers.
        text = self.complete(messages, **kwargs)
        if text:
            yield text

    def stop_stream(self) -> None:
        if hasattr(self.provider, "stop_stream"):
            self.provider.stop_stream()

    def _load_provider(self, provider_name: str) -> Any:
        provider_file = resolve_code_root(self.repo_root) / "LLMProviders" / provider_name / "provider.py"
        if not provider_file.is_file():
            raise FileNotFoundError(f"Provider file not found: {provider_file}")

        module_name = f"aidzero_provider_{provider_name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, provider_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load provider module: {provider_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        provider_cls = self._find_provider_class(module)
        return provider_cls()

    @staticmethod
    def _find_provider_class(module: Any) -> type[Any]:
        candidates: list[type[Any]] = []
        for _, member in inspect.getmembers(module, inspect.isclass):
            if member.__module__ != module.__name__:
                continue
            if not member.__name__.endswith("Provider"):
                continue
            if member.__name__ == "OpenAICompatibleProvider":
                continue
            candidates.append(member)

        if not candidates:
            raise RuntimeError(f"No provider class found in '{module.__name__}'.")
        candidates.sort(key=lambda item: item.__name__)
        return candidates[0]

    @staticmethod
    def _flatten_messages(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)

    def _extract_text(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload.strip()
        if not isinstance(payload, dict):
            return str(payload).strip()

        text = self._extract_openai(payload)
        if text:
            return text
        text = self._extract_claude(payload)
        if text:
            return text
        text = self._extract_gemini(payload)
        if text:
            return text

        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _extract_openai(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""

        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        texts.append(part["text"])
                if texts:
                    return "\n".join(texts).strip()
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                serialized = LLMClient._serialize_tool_call_list(tool_calls)
                if serialized:
                    return serialized

        content = first.get("text")
        if isinstance(content, str):
            return content.strip()
        return ""

    @staticmethod
    def _extract_claude(payload: dict[str, Any]) -> str:
        content = payload.get("content")
        if not isinstance(content, list):
            return ""
        chunks: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                chunks.append(block["text"])
        return "\n".join(chunks).strip()

    @staticmethod
    def _extract_gemini(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            return ""
        texts: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
        return "\n".join(texts).strip()

    @staticmethod
    def _extract_openai_stream(payload: Any) -> list[str]:
        if not isinstance(payload, dict):
            text = str(payload).strip()
            return [text] if text else []

        chunks: list[str] = []
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return chunks
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            chunks.append(part["text"])
                        elif isinstance(part, str):
                            chunks.append(part)
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            chunks.append(part["text"])
                        elif isinstance(part, str):
                            chunks.append(part)
            text = choice.get("text")
            if isinstance(text, str):
                chunks.append(text)
        return chunks

    @staticmethod
    def _serialize_tool_call_list(tool_calls: list[Any]) -> str:
        for raw_call in tool_calls:
            if not isinstance(raw_call, dict):
                continue
            function_payload = raw_call.get("function")
            if not isinstance(function_payload, dict):
                continue
            name = function_payload.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            arguments_raw = function_payload.get("arguments", "{}")
            arguments_value: Any
            if isinstance(arguments_raw, str):
                text = arguments_raw.strip()
                if not text:
                    arguments_value = {}
                else:
                    try:
                        arguments_value = json.loads(text)
                    except json.JSONDecodeError:
                        arguments_value = {"raw": arguments_raw}
            elif isinstance(arguments_raw, dict):
                arguments_value = arguments_raw
            else:
                arguments_value = {"raw": arguments_raw}
            payload = {"name": name.strip(), "arguments": arguments_value}
            return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"
        return ""

    @staticmethod
    def _collect_openai_stream_tool_calls(
        payload: Any,
        collector: dict[int, dict[str, str]],
    ) -> None:
        if not isinstance(payload, dict):
            return
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                raw_tool_calls = delta.get("tool_calls")
                if isinstance(raw_tool_calls, list):
                    LLMClient._merge_tool_call_chunks(raw_tool_calls, collector)
            message = choice.get("message")
            if isinstance(message, dict):
                raw_tool_calls = message.get("tool_calls")
                if isinstance(raw_tool_calls, list):
                    LLMClient._merge_tool_call_chunks(raw_tool_calls, collector)

    @staticmethod
    def _merge_tool_call_chunks(
        tool_calls: list[Any],
        collector: dict[int, dict[str, str]],
    ) -> None:
        for order, raw_call in enumerate(tool_calls):
            if not isinstance(raw_call, dict):
                continue
            raw_index = raw_call.get("index")
            if isinstance(raw_index, int):
                index = raw_index
            else:
                index = order
            slot = collector.setdefault(index, {"name": "", "arguments": ""})
            function_payload = raw_call.get("function")
            if not isinstance(function_payload, dict):
                continue
            raw_name = function_payload.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                slot["name"] = raw_name.strip()
            raw_arguments = function_payload.get("arguments")
            if isinstance(raw_arguments, str):
                slot["arguments"] += raw_arguments
            elif isinstance(raw_arguments, dict):
                slot["arguments"] += json.dumps(raw_arguments, ensure_ascii=False)

    @staticmethod
    def _serialize_first_tool_call(collector: dict[int, dict[str, str]]) -> str:
        for index in sorted(collector.keys()):
            slot = collector[index]
            name = slot.get("name", "").strip()
            if not name:
                continue
            raw_arguments = slot.get("arguments", "").strip()
            if not raw_arguments:
                arguments_value: Any = {}
            else:
                try:
                    arguments_value = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments_value = {"raw": raw_arguments}
            payload = {"name": name, "arguments": arguments_value}
            return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"
        return ""
