from __future__ import annotations

from DASH import new


class _AppWithNewConversation:
    def __init__(self) -> None:
        self.called = 0

    def start_new_conversation(self) -> None:
        self.called += 1


class _FallbackApp:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def _append_system_line(self, text: str) -> None:
        self.lines.append(text)


def test_new_command_calls_app_session_reset():
    app = _AppWithNewConversation()

    handled = new.run("/new", app=app)

    assert handled is True
    assert app.called == 1


def test_new_command_fallback_message_when_ui_has_no_handler():
    app = _FallbackApp()

    handled = new.run("/new", app=app)

    assert handled is True
    assert app.lines == ["This UI does not support /new."]


def test_new_command_match_is_case_insensitive_and_trimmed():
    assert new.match("/new") is True
    assert new.match("  /NEW  ") is True
    assert new.match("/new now") is False
