from __future__ import annotations

import pytest

from TOOLS.sandbox_run import run as sandbox_run


def test_sandbox_run_executes_shell_command(tmp_path):
    result = sandbox_run(
        {"command": "printf hello", "timeout_seconds": 5},
        repo_root=tmp_path,
        memory=None,
    )

    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]


def test_sandbox_run_requires_command(tmp_path):
    with pytest.raises(ValueError, match="'command' is required"):
        sandbox_run(
            {},
            repo_root=tmp_path,
            memory=None,
        )
