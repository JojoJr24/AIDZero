from __future__ import annotations

from core.models import TriggerEvent
from UI.terminal.app import TerminalApp


class _FailingEngine:
    def run_event(self, event):
        del event
        raise RuntimeError("provider offline")


class _Gateway:
    def collect(self, *, trigger: str, prompt: str):
        del trigger, prompt
        return [TriggerEvent(kind="interactive", source="terminal", prompt="hola")]


class _History:
    def add_prompt(self, prompt: str):
        del prompt
        return []

    def list_prompts(self, *, limit: int | None = None):
        del limit
        return []


def test_terminal_app_shows_provider_error_without_crashing(tmp_path, capsys):
    app = TerminalApp(
        repo_root=tmp_path,
        engine=_FailingEngine(),
        gateway=_Gateway(),
        history=_History(),
    )

    exit_code = app.run(request="hola", trigger="interactive")

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Provider/core error: provider offline" in captured.out
