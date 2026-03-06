from __future__ import annotations

import asyncio
from pathlib import Path

from UI.tui.textual_app import _TextualTUI, _read_bool_env


def test_read_bool_env_uses_default_when_missing(monkeypatch):
    monkeypatch.delenv("AIDZERO_TUI_MOUSE", raising=False)
    assert _read_bool_env("AIDZERO_TUI_MOUSE", default=True) is True
    assert _read_bool_env("AIDZERO_TUI_MOUSE", default=False) is False


def test_read_bool_env_accepts_true_values(monkeypatch):
    for value in ("1", "true", "yes", "on", " TRUE "):
        monkeypatch.setenv("AIDZERO_TUI_MOUSE", value)
        assert _read_bool_env("AIDZERO_TUI_MOUSE", default=False) is True


def test_read_bool_env_accepts_false_values(monkeypatch):
    for value in ("0", "false", "no", "off", " OFF "):
        monkeypatch.setenv("AIDZERO_TUI_MOUSE", value)
        assert _read_bool_env("AIDZERO_TUI_MOUSE", default=True) is False


def test_read_bool_env_falls_back_to_default_on_invalid_value(monkeypatch):
    monkeypatch.setenv("AIDZERO_TUI_MOUSE", "maybe")
    assert _read_bool_env("AIDZERO_TUI_MOUSE", default=True) is True
    assert _read_bool_env("AIDZERO_TUI_MOUSE", default=False) is False


def test_mark_idle_from_worker_falls_back_when_thread_handoff_fails():
    app = _TextualTUI.__new__(_TextualTUI)
    app._busy = True
    app._stop_requested = True

    def _raise_call_from_thread(*_args, **_kwargs):
        raise RuntimeError("app not running")

    app.call_from_thread = _raise_call_from_thread

    app._mark_idle_from_worker()

    assert app._busy is False
    assert app._stop_requested is False


def test_process_prompt_worker_marks_idle_on_unexpected_error():
    app = _TextualTUI.__new__(_TextualTUI)
    app._busy = True
    app._stop_requested = False

    class _BrokenGateway:
        def collect(self, *, trigger: str, prompt: str):
            del trigger, prompt
            raise RuntimeError("boom")

    app.gateway = _BrokenGateway()
    messages: list[str] = []
    idle_calls: list[None] = []
    app._append_system_line = messages.append
    app.call_from_thread = lambda callback, *args: callback(*args)
    app._mark_idle_from_worker = lambda: idle_calls.append(None)

    app._process_prompt_worker("test prompt")

    assert messages == ["Unexpected worker error: boom"]
    assert len(idle_calls) == 1


def test_queue_prompt_recovers_when_worker_start_fails():
    app = _TextualTUI.__new__(_TextualTUI)
    app._busy = False
    app._stop_requested = False

    class _History:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def add_prompt(self, prompt: str) -> None:
            self.prompts.append(prompt)

    history = _History()
    app.history = history
    user_prompts: list[str] = []
    system_lines: list[str] = []
    app._append_user_prompt = user_prompts.append
    app._append_system_line = system_lines.append

    def _failing_run_worker(*_args, **_kwargs):
        raise RuntimeError("worker start failed")

    app.run_worker = _failing_run_worker

    app._queue_prompt("hello")

    assert user_prompts == ["hello"]
    assert history.prompts == ["hello"]
    assert app._busy is False
    assert app._stop_requested is False
    assert system_lines == ["Could not start request worker: worker start failed"]


def test_build_status_text_reports_runtime_flags():
    app = _TextualTUI.__new__(_TextualTUI)
    app.agent_profile = type("Profile", (), {"name": "default", "description": ""})()
    app.active_trigger = "interactive"
    app._busy = True
    app._stop_requested = False
    app.history = type("History", (), {"enabled": False})()
    app.engine = type("Engine", (), {"memory_store": type("MemoryStore", (), {"enabled": True})()})()

    status = app._build_status_text()

    assert "busy" in status
    assert "history:off" in status
    assert "memory:on" in status
    assert "interactive" not in status


def test_build_header_text_uses_agent_name():
    app = _TextualTUI.__new__(_TextualTUI)
    app.agent_profile = type("Profile", (), {"name": "builder", "description": "Focused coding profile"})()

    assert app._build_header_text() == "AIDZero TUI | agent builder"


def test_build_welcome_text_lists_shortcuts_and_commands():
    app = _TextualTUI.__new__(_TextualTUI)
    app.agent_profile = type("Profile", (), {"name": "builder", "description": ""})()

    class _DashCommands:
        def suggestions(self, query: str):
            assert query == "/"
            return [
                ("/agent", "Switch agent"),
                ("/history", "Show recent prompts"),
                ("/new", "Start a new conversation"),
            ]

    app.dash_commands = _DashCommands()

    welcome = app._build_welcome_text()

    assert "Ready. Type a prompt or / to browse commands." in welcome
    assert "Ctrl+R history" in welcome
    assert "Commands: /agent, /history, /new" in welcome


def test_overlay_windows_are_centered():
    class _MemoryStore:
        enabled = True

    class _LLM:
        pass

    class _Engine:
        memory_store = _MemoryStore()
        llm = _LLM()
        history_enabled = True

        def reset_session(self) -> None:
            return None

    class _Gateway:
        def collect(self, *, trigger: str, prompt: str):
            del trigger, prompt
            return []

    class _History:
        enabled = True

        def list_prompts(self, *, limit: int = 30):
            del limit
            return ["one", "two"]

        def add_prompt(self, prompt: str) -> None:
            del prompt

    class _Manager:
        is_remote = False

    class _Profile:
        name = "default"
        description = ""
        enabled_dash_modules: list[str] = []
        enabled_tools: list[str] = []
        system_prompt = ""
        memory_enabled = True
        history_enabled = True

    async def _run() -> None:
        app = _TextualTUI(
            repo_root=Path("/mnt/ssd_storage/IA/AutoAgent"),
            engine=_Engine(),
            gateway=_Gateway(),
            history=_History(),
            agent_manager=_Manager(),
            agent_profile=_Profile(),
            trigger="interactive",
            initial_request=None,
        )
        async with app.run_test(size=(120, 40)) as pilot:
            app._show_overlay("#command-panel")
            await pilot.pause()
            panel = app.query_one("#command-panel")
            assert panel.region.x == 4
            assert panel.region.y == 18

            app._show_history_selector(limit=30)
            await pilot.pause()
            history_panel = app.query_one("#history-panel")
            assert history_panel.region.x == 4
            assert history_panel.region.y == 17

    asyncio.run(_run())
