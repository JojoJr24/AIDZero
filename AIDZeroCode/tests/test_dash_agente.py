from __future__ import annotations

from DASH import agente as agent


class _Profile:
    def __init__(
        self,
        name: str,
        enabled_tools=None,
        enabled_dash_modules=None,
        *,
        memory_enabled: bool = True,
        history_enabled: bool = True,
    ) -> None:
        self.name = name
        self.enabled_tools = enabled_tools
        self.enabled_dash_modules = enabled_dash_modules
        self.memory_enabled = memory_enabled
        self.history_enabled = history_enabled


class _Manager:
    def __init__(self) -> None:
        self._names = ["default", "planificador"]

    def list_profile_names(self) -> list[str]:
        return self._names


class _App:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.agent_manager = _Manager()
        self.agent_profile = _Profile("default")
        self.switched: str | None = None

    def _append_system_line(self, text: str) -> None:
        self.lines.append(text)

    def switch_agent_profile(self, name: str):
        self.switched = name
        self.agent_profile = _Profile(name, enabled_tools=["read_text"], enabled_dash_modules=None)
        return self.agent_profile


def test_agente_lists_profiles():
    app = _App()

    handled = agent.run("/agent", app=app)

    assert handled is True
    assert any("default (activo)" in line for line in app.lines)
    assert any("planificador" in line for line in app.lines)


def test_agente_switches_profile():
    app = _App()

    handled = agent.run("/agent planificador", app=app)

    assert handled is True
    assert app.switched == "planificador"
    assert any("agent activo: planificador" in line for line in app.lines)
    assert any("- memory: on" in line for line in app.lines)
    assert any("- history: on" in line for line in app.lines)
