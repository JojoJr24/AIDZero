from __future__ import annotations

from agent.ui_registry import UIRegistry


def test_ui_registry_discovers_py_modules_and_runs_ui(tmp_path):
    ui_root = tmp_path / "UI"
    ui_root.mkdir(parents=True, exist_ok=True)

    (ui_root / "terminal.py").write_text(
        "\n".join(
            [
                "def run_ui(**kwargs):",
                "    assert kwargs['provider_name'] == 'openai'",
                "    return 7",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (ui_root / "_private.py").write_text("def run_ui(**kwargs): return 1\n", encoding="utf-8")

    registry = UIRegistry(tmp_path)
    assert registry.names() == ["terminal"]

    exit_code = registry.run("terminal", provider_name="openai")
    assert exit_code == 7
