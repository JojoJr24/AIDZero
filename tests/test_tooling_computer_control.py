from __future__ import annotations

import shutil

import pytest

from TOOLS.computer_control import run as computer_control_run


def test_computer_control_run_executes_shell_command(tmp_path):
    result = computer_control_run(
        {"action": "run", "command": "printf hello", "timeout_seconds": 5},
        repo_root=tmp_path,
        memory=None,
    )

    assert result["status"] == "ok"
    assert result["action"] == "run"
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]


def test_computer_control_rejects_unknown_action(tmp_path):
    with pytest.raises(ValueError, match="Unsupported computer_control action"):
        computer_control_run(
            {"action": "teleport"},
            repo_root=tmp_path,
            memory=None,
        )


def test_computer_control_screenshot_fails_without_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="No screenshot backend"):
        computer_control_run(
            {"action": "screenshot"},
            repo_root=tmp_path,
            memory=None,
        )
