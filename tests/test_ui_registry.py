from __future__ import annotations

from agent.ui_registry import UIRegistry


def test_ui_registry_discovers_ui_folders_and_runs_entrypoint(tmp_path):
    ui_root = tmp_path / "UI"
    ui_root.mkdir(parents=True, exist_ok=True)

    terminal_dir = ui_root / "terminal"
    terminal_dir.mkdir(parents=True, exist_ok=True)
    (terminal_dir / "entrypoint.py").write_text(
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

    private_dir = ui_root / "_private"
    private_dir.mkdir(parents=True, exist_ok=True)
    (private_dir / "entrypoint.py").write_text("def run_ui(**kwargs): return 1\n", encoding="utf-8")

    registry = UIRegistry(tmp_path)
    assert registry.names() == ["terminal"]

    exit_code = registry.run("terminal", provider_name="openai")
    assert exit_code == 7
