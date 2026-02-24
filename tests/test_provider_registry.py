"""Tests for provider registry configuration."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.provider_registry import ProviderRegistry


def test_provider_registry_includes_claude() -> None:
    registry = ProviderRegistry(REPO_ROOT)
    assert registry.has("AID-claude")
    assert "AID-claude" in registry.names()
    assert registry.default_model("AID-claude") == "claude-3-5-sonnet-latest"
