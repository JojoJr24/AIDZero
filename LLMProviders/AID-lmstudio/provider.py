"""LM Studio provider adapter via OpenAI-compatible endpoints."""

from __future__ import annotations

import os

from LLMProviders.openai_compatible_provider import (
    OpenAICompatibleProvider,
    OpenAICompatibleProviderError,
    extract_text_from_chat_completion,
)

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"


class LMStudioProviderError(OpenAICompatibleProviderError):
    """Raised when the LM Studio provider cannot complete a request."""


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio adapter using OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
    ) -> None:
        resolved_base_url = (base_url or os.getenv("LMSTUDIO_BASE_URL", DEFAULT_BASE_URL)).strip()
        if not resolved_base_url:
            resolved_base_url = DEFAULT_BASE_URL
        super().__init__(
            provider_label="LM Studio",
            base_url=resolved_base_url,
            api_key=api_key,
            api_key_env="LMSTUDIO_API_KEY",
            require_api_key=False,
            timeout=timeout,
            error_cls=LMStudioProviderError,
        )


__all__ = ["LMStudioProvider", "LMStudioProviderError", "extract_text_from_chat_completion"]
