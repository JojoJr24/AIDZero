from __future__ import annotations

from DASH import load


class _FakeApp:
    def __init__(self, repo_root) -> None:
        self.repo_root = repo_root
        self.lines: list[str] = []

    def _append_system_line(self, text: str) -> None:
        self.lines.append(text)


def test_load_returns_text_to_prefill_input(tmp_path):
    app = _FakeApp(tmp_path)
    result = load.run("/load revisar estado del repo", app=app)
    assert result == "revisar estado del repo"


def test_load_returns_usage_on_invalid_command(tmp_path):
    app = _FakeApp(tmp_path)
    result = load.run("/load", app=app)
    assert result is True
    assert app.lines[-1] == "Usage: /load <text>"
