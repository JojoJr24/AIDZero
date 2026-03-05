from __future__ import annotations

from DASH import agente


class _AppNoManager:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def _append_system_line(self, text: str) -> None:
        self.lines.append(text)


def test_agent_command_handles_ui_without_profile_manager():
    app = _AppNoManager()

    handled = agente.run("/agent", app=app)

    assert handled is True
    assert app.lines == ["Este UI no soporta perfiles de agent."]
