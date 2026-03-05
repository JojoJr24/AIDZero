from __future__ import annotations

from pathlib import Path

from AIDZero import _list_provider_models


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_list_provider_models_uses_list_model_names(tmp_path: Path) -> None:
    _write(
        tmp_path / "LLMProviders" / "demo" / "provider.py",
        "class DemoProvider:\n"
        "    def list_model_names(self):\n"
        "        return ['model-a', 'model-b', 'model-a']\n",
    )

    models = _list_provider_models(tmp_path, "demo")

    assert models == ["model-a", "model-b"]
