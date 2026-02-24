"""Shared provider contracts used by AIDZero provider adapters."""

from __future__ import annotations

from typing import Any, Iterator, Protocol


class ProviderError(RuntimeError):
    """Base error for provider operations."""


class LLMProvider(Protocol):
    """Uniform contract all provider adapters should implement."""

    def list_models(self, *, page_size: int = 100) -> list[dict[str, Any]]:
        """Return available models from the provider."""

    def list_model_names(self, *, page_size: int = 100) -> list[str]:
        """Return only model names for quick selection."""

    def generate_content(self, model: str, contents: Any, **kwargs: Any) -> dict[str, Any]:
        """Run a non-streaming content generation request."""

    def stream_generate_content(
        self, model: str, contents: Any, **kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        """Run a streaming content generation request."""

    def generate_text(self, model: str, prompt: str, **kwargs: Any) -> str:
        """Run a simple text completion style request."""

    def stream_generate_text(self, model: str, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Run a streaming text completion style request."""

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Run a chat request based on message history."""

    def stream_chat(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        """Run a streaming chat request based on message history."""

    def count_tokens(self, model: str, contents: Any, **kwargs: Any) -> dict[str, Any]:
        """Count tokens for a request payload."""

    def embed_content(self, model: str, content: Any, **kwargs: Any) -> dict[str, Any]:
        """Generate an embedding for one content input."""
