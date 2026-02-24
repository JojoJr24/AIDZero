"""llama.cpp provider adapter via OpenAI-compatible endpoints."""

from __future__ import annotations

import os

from agent.openai_compatible_provider import (
    OpenAICompatibleProvider,
    OpenAICompatibleProviderError,
    extract_text_from_chat_completion,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


class LlamaCppProviderError(OpenAICompatibleProviderError):
    """Raised when the llama.cpp provider cannot complete a request."""


class LlamaCppProvider(OpenAICompatibleProvider):
    """llama.cpp adapter using OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 480,
    ) -> None:
        resolved_base_url = (base_url or os.getenv("LLAMACPP_BASE_URL", DEFAULT_BASE_URL)).strip()
        if not resolved_base_url:
            resolved_base_url = DEFAULT_BASE_URL
        super().__init__(
            provider_label="llama.cpp",
            base_url=resolved_base_url,
            api_key=api_key,
            api_key_env="LLAMACPP_API_KEY",
            require_api_key=False,
            timeout=timeout,
            error_cls=LlamaCppProviderError,
        )


__all__ = ["LlamaCppProvider", "LlamaCppProviderError", "extract_text_from_chat_completion"]
