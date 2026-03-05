from __future__ import annotations

from DASH import history


class _HistoryStore:
    def __init__(self, prompts: list[str]) -> None:
        self._prompts = prompts

    def list_prompts(self, *, limit: int | None = None):
        if limit is None:
            return list(self._prompts)
        return list(self._prompts)[:limit]


class _FakeApp:
    def __init__(self, prompts: list[str]) -> None:
        self.history = _HistoryStore(prompts)
        self.lines: list[str] = []

    def _append_system_line(self, text: str) -> None:
        self.lines.append(text)


def test_history_command_prints_recent_prompts_when_selector_unavailable():
    app = _FakeApp(["one", "two"])

    handled = history.run("/history", app=app)

    assert handled is True
    assert app.lines[0] == "Recent prompts:"
    assert "one" in app.lines[1]
    assert "two" in app.lines[2]


def test_history_command_handles_empty_history():
    app = _FakeApp([])

    handled = history.run("/history", app=app)

    assert handled is True
    assert app.lines == ["History is empty."]
