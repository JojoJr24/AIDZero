"""Tests for dynamic UI registry and UI option parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AIDZero import _parse_ui_options
from agent.ui_registry import UIRegistry


def test_ui_registry_discovers_runtime_uis() -> None:
    registry = UIRegistry(REPO_ROOT)
    names = registry.names()
    assert "terminal" in names
    assert "web" in names


def test_ui_option_parser_accepts_key_value() -> None:
    parsed = _parse_ui_options(["host=127.0.0.1", "port=8787", "empty="])
    assert parsed == {"host": "127.0.0.1", "port": "8787", "empty": ""}


def test_ui_option_parser_rejects_invalid_items() -> None:
    with pytest.raises(ValueError):
        _parse_ui_options(["invalid"])
    with pytest.raises(ValueError):
        _parse_ui_options(["=missing_key"])
