"""Ollama provider adapter via OpenAI-compatible endpoints."""

from __future__ import annotations

import os

from agent.openai_compatible_provider import (
    OpenAICompatibleProvider,
    OpenAICompatibleProviderError,
    extract_text_from_chat_completion,
)

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"


class OllamaProviderError(OpenAICompatibleProviderError):
    """Raised when the Ollama provider cannot complete a request."""


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama adapter using its OpenAI-compatible API surface."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
    ) -> None:
        resolved_base_url = (base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)).strip()
        if not resolved_base_url:
            resolved_base_url = DEFAULT_BASE_URL
        super().__init__(
            provider_label="Ollama",
            base_url=resolved_base_url,
            api_key=api_key,
            api_key_env="OLLAMA_API_KEY",
            require_api_key=False,
            timeout=timeout,
            error_cls=OllamaProviderError,
        )


__all__ = ["OllamaProvider", "OllamaProviderError", "extract_text_from_chat_completion"]
