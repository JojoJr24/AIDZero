"""UI display naming normalization helpers."""

from __future__ import annotations


def to_ui_label(value: str) -> str:
    """Normalize non-model runtime labels for display surfaces."""
    normalized = value.strip()
    if not normalized:
        return normalized
    if normalized.startswith("AID-"):
        return normalized.removeprefix("AID-")
    return f"{normalized} (test)"


def to_ui_model_label(value: str) -> str:
    """Model labels are displayed exactly as provided."""
    return value.strip()
