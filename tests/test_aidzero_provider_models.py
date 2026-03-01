from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from AIDZero import _list_provider_models, _pick_model_for_provider


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


def test_pick_model_for_provider_falls_back_to_manual_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("AIDZero._list_provider_models", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    answers = iter(["custom-model"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    selected = _pick_model_for_provider(Path("."), "demo")

    assert selected == "custom-model"
