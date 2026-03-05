"""Shared models for the runtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class TriggerEvent:
    """One unit of work emitted by the gateway."""

    kind: str
    source: str
    prompt: str
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSchema:
    """Tool schema injected into the LLM each turn."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool call extracted from model output."""

    name: str
    arguments: dict[str, Any]
    raw_block: str


@dataclass(frozen=True)
class TurnResult:
    """Final outcome of one gateway event."""

    event: TriggerEvent
    response: str
    rounds: int
    used_tools: list[str]


@dataclass(frozen=True)
class RuntimeConfig:
    """User-selected runtime defaults."""

    ui: str
    provider: str
    model: str


@dataclass(frozen=True)
class ToolExecutionResult:
    """Structured tool response fed back into the model."""

    tool_name: str
    status: str
    payload: Any
