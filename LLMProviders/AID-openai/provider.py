"""OpenAI provider adapter built on the shared OpenAI-compatible transport."""

from __future__ import annotations

import os

from agent.openai_compatible_provider import (
    OpenAICompatibleProvider,
    OpenAICompatibleProviderError,
    extract_text_from_chat_completion,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProviderError(OpenAICompatibleProviderError):
    """Raised when the OpenAI provider cannot complete a request."""


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider implementation using OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
    ) -> None:
        resolved_base_url = (base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)).strip()
        if not resolved_base_url:
            resolved_base_url = DEFAULT_BASE_URL
        super().__init__(
            provider_label="OpenAI",
            base_url=resolved_base_url,
            api_key=api_key,
            api_key_env="OPENAI_API_KEY",
            require_api_key=True,
            timeout=timeout,
            error_cls=OpenAIProviderError,
        )


__all__ = ["OpenAIProvider", "OpenAIProviderError", "extract_text_from_chat_completion"]
