from __future__ import annotations

import json

import pytest

from CORE.ui_registry import UIRegistry


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


def test_ui_registry_discovers_from_nested_code_root(tmp_path):
    ui_root = tmp_path / "AIDZeroCode" / "UI"
    ui_root.mkdir(parents=True, exist_ok=True)

    terminal_dir = ui_root / "terminal"
    terminal_dir.mkdir(parents=True, exist_ok=True)
    (terminal_dir / "entrypoint.py").write_text(
        "\n".join(
            [
                "def run_ui(**kwargs):",
                "    del kwargs",
                "    return 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    registry = UIRegistry(tmp_path)
    assert registry.names() == ["terminal"]


def test_ui_registry_discovers_thirdparty_ui_without_entrypoint(tmp_path):
    ui_root = tmp_path / "UI"
    ui_root.mkdir(parents=True, exist_ok=True)

    android_dir = ui_root / "AndroidApp"
    android_dir.mkdir(parents=True, exist_ok=True)
    (android_dir / "ui.json").write_text(
        json.dumps({"type": "thirdparty"}, ensure_ascii=False),
        encoding="utf-8",
    )

    registry = UIRegistry(tmp_path)

    assert registry.names() == ["AndroidApp"]
    assert registry.ui_type("AndroidApp") == "thirdparty"
    with pytest.raises(RuntimeError, match="thirdparty"):
        registry.run("AndroidApp")
