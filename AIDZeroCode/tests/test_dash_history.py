from __future__ import annotations

from DASH import history


class _HistoryStore:
    def __init__(self, prompts: list[str]) -> None:
        self._prompts = prompts

    def list_prompts(self, *, limit: int | None = None) -> list[str]:
        if limit is None:
            return list(self._prompts)
        return list(self._prompts[:limit])


class _AppWithSelector:
    def __init__(self) -> None:
        self.called_with: int | None = None

    def _show_history_selector(self, *, limit: int = 30) -> bool:
        self.called_with = limit
        return True


class _FallbackApp:
    def __init__(self, prompts: list[str]) -> None:
        self.history = _HistoryStore(prompts)
        self.lines: list[str] = []

    def _append_system_line(self, text: str) -> None:
        self.lines.append(text)


def test_history_command_opens_selector_when_available():
    app = _AppWithSelector()

    handled = history.run("/history", app=app)

    assert handled is True
    assert app.called_with == 30


def test_history_command_fallback_prints_history_when_selector_missing():
    app = _FallbackApp(["p1", "p2"])

    handled = history.run("/history", app=app)

    assert handled is True
    assert app.lines == ["Recent prompts:", " 1. p1", " 2. p2"]

