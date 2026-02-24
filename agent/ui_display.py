"""Shared display helpers for UI-facing labels."""

from __future__ import annotations

AID_PREFIX = "AID-"
TEST_SUFFIX = "(test)"


def to_ui_label(value: str) -> str:
    """Convert internal names to user-facing labels for all UIs."""
    normalized = value.strip()
    if not normalized:
        return normalized

    if normalized.startswith(AID_PREFIX):
        trimmed = normalized[len(AID_PREFIX) :].strip()
        return trimmed or normalized

    if normalized.endswith(TEST_SUFFIX):
        return normalized
    return f"{normalized} {TEST_SUFFIX}"


def to_ui_model_label(value: str) -> str:
    """Model names must be shown as-is in the UI."""
    return value.strip()
