"""Slash command to list/switch active agent profiles from Agents/*.json."""

from __future__ import annotations

DASH_COMMANDS = [
    {"command": "/agent", "description": "Lista agentes disponibles y activo"},
    {"command": "/agent <nombre>", "description": "Cambia el agent activo"},
    {"command": "/agent list", "description": "Lista agentes disponibles"},
]


def match(raw: str) -> bool:
    text = raw.strip().lower()
    return text == "/agent" or text.startswith("/agent ")


def run(raw: str, *, app) -> bool:
    text = raw.strip()
    parts = text.split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else ""

    manager = getattr(app, "agent_manager", None)
    profile = getattr(app, "agent_profile", None)

    if manager is None:
        _line(app, "Este UI no soporta perfiles de agent.")
        return True

    if not arg or arg.lower() in {"list", "ls"}:
        names = manager.list_profile_names()
        active_name = getattr(profile, "name", None)
        if not names:
            _line(app, "No hay perfiles en Agents/*.json")
            return True
        _line(app, "Agentes disponibles:")
        for name in names:
            marker = " (activo)" if name == active_name else ""
            _line(app, f"- {name}{marker}")
        _line(app, "Uso: /agent <nombre>")
        return True

    target = arg.split()[0].strip()
    if not target:
        _line(app, "Uso: /agent <nombre>")
        return True

    switcher = getattr(app, "switch_agent_profile", None)
    if not callable(switcher):
        _line(app, "Este UI no permite cambiar de agent en caliente.")
        return True

    try:
        selected = switcher(target)
    except Exception as error:  # noqa: BLE001
        _line(app, f"No se pudo activar '{target}': {error}")
        return True

    tools = ", ".join(selected.enabled_tools) if selected.enabled_tools else "all"
    dash = ", ".join(selected.enabled_dash_modules) if selected.enabled_dash_modules else "all"
    memory = "on" if getattr(selected, "memory_enabled", True) else "off"
    history = "on" if getattr(selected, "history_enabled", True) else "off"
    _line(app, f"agent activo: {selected.name}")
    _line(app, f"- tools: {tools}")
    _line(app, f"- dash: {dash}")
    _line(app, f"- memory: {memory}")
    _line(app, f"- history: {history}")
    return True


def _line(app, text: str) -> None:
    writer = getattr(app, "_append_system_line", None)
    if callable(writer):
        writer(text)
