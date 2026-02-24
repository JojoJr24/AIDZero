"""Tests for UI display label normalization."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.ui_display import to_ui_label, to_ui_model_label


def test_to_ui_label_removes_aid_prefix() -> None:
    assert to_ui_label("AID-claude") == "claude"
    assert to_ui_label("AID-google_gemini") == "google_gemini"


def test_to_ui_label_appends_test_when_no_prefix() -> None:
    assert to_ui_label("terminal") == "terminal (test)"


def test_to_ui_label_does_not_duplicate_suffix() -> None:
    assert to_ui_label("custom (test)") == "custom (test)"


def test_to_ui_model_label_keeps_model_raw() -> None:
    assert to_ui_model_label("gpt-4o-mini") == "gpt-4o-mini"
    assert to_ui_model_label(" claude-3-5-sonnet-latest ") == "claude-3-5-sonnet-latest"
