from __future__ import annotations

from CORE.dash_commands import build_default_dash_command_registry


class _FakeApp:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.input_text: str = ""

    def _set_input_text(self, text: str) -> None:
        self.input_text = text


def test_dash_registry_loads_modules_from_dash_folder(tmp_path):
    dash_root = tmp_path / "DASH"
    dash_root.mkdir(parents=True, exist_ok=True)

    (dash_root / "hello.py").write_text(
        "\n".join(
            [
                "DASH_COMMANDS = [",
                "    {'command': '/hello', 'description': 'Say hello'},",
                "]",
                "",
                "def match(raw: str) -> bool:",
                "    return raw == '/hello'",
                "",
                "def run(raw: str, *, app):",
                "    del raw",
                "    app.calls.append('hello')",
                "    return True",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (dash_root / "bye.py").write_text(
        "\n".join(
            [
                "DASH_COMMANDS = [",
                "    {'command': '/bye', 'description': 'Say bye'},",
                "]",
                "",
                "def match(raw: str) -> bool:",
                "    return raw == '/bye'",
                "",
                "def run(raw: str, *, app):",
                "    del raw",
                "    app.calls.append('bye')",
                "    return True",
                "",
            ]
        ),
        encoding="utf-8",
    )

    registry = build_default_dash_command_registry(tmp_path)
    app = _FakeApp()

    assert registry.suggestions("/") == [("/bye", "Say bye"), ("/hello", "Say hello")]
    assert registry.suggestions("/he") == [("/hello", "Say hello")]
    assert registry.handle("/hello", app=app) is True
    assert app.calls == ["hello"]


def test_dash_registry_loads_modules_from_nested_code_root(tmp_path):
    dash_root = tmp_path / "AIDZeroCode" / "DASH"
    dash_root.mkdir(parents=True, exist_ok=True)

    (dash_root / "hello.py").write_text(
        "\n".join(
            [
                "DASH_COMMANDS = [",
                "    {'command': '/hello', 'description': 'Say hello'},",
                "]",
                "",
                "def match(raw: str) -> bool:",
                "    return raw == '/hello'",
                "",
                "def run(raw: str, *, app):",
                "    del raw, app",
                "    return True",
                "",
            ]
        ),
        encoding="utf-8",
    )

    registry = build_default_dash_command_registry(tmp_path)
    assert registry.suggestions("/") == [("/hello", "Say hello")]


def test_dash_registry_picks_up_new_file_without_code_changes(tmp_path):
    dash_root = tmp_path / "DASH"
    dash_root.mkdir(parents=True, exist_ok=True)

    (dash_root / "alpha.py").write_text(
        "\n".join(
            [
                "DASH_COMMANDS = [",
                "    {'command': '/alpha', 'description': 'Alpha command'},",
                "]",
                "",
                "def match(raw: str) -> bool:",
                "    return raw == '/alpha'",
                "",
                "def run(raw: str, *, app):",
                "    app.calls.append('alpha')",
                "    return True",
                "",
            ]
        ),
        encoding="utf-8",
    )
    registry = build_default_dash_command_registry(tmp_path)
    assert registry.suggestions("/") == [("/alpha", "Alpha command")]

    (dash_root / "zeta.py").write_text(
        "\n".join(
            [
                "DASH_COMMANDS = [",
                "    {'command': '/zeta', 'description': 'Zeta command'},",
                "]",
                "",
                "def match(raw: str) -> bool:",
                "    return raw == '/zeta'",
                "",
                "def run(raw: str, *, app):",
                "    app.calls.append('zeta')",
                "    return True",
                "",
            ]
        ),
        encoding="utf-8",
    )

    reloaded = build_default_dash_command_registry(tmp_path)
    assert reloaded.suggestions("/") == [("/alpha", "Alpha command"), ("/zeta", "Zeta command")]


def test_dash_registry_writes_string_result_to_input(tmp_path):
    dash_root = tmp_path / "DASH"
    dash_root.mkdir(parents=True, exist_ok=True)

    (dash_root / "hola.py").write_text(
        "\n".join(
            [
                "DASH_COMMANDS = [",
                "    {'command': '/hola', 'description': 'Escribe hola en input'},",
                "]",
                "",
                "def match(raw: str) -> bool:",
                "    return raw == '/hola'",
                "",
                "def run(raw: str, *, app):",
                "    del raw, app",
                "    return 'Hola'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    registry = build_default_dash_command_registry(tmp_path)
    app = _FakeApp()
    assert registry.handle("/hola", app=app) is True
    assert app.input_text == "Hola"
