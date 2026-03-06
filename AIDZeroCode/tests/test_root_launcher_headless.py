from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_launcher_module():
    module_path = Path(__file__).resolve().parents[2] / "AIDZero.py"
    spec = importlib.util.spec_from_file_location("aidzero_root_launcher_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load launcher module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_headless_prompt_read_and_result_write(tmp_path):
    launcher = _load_launcher_module()
    repo_root = tmp_path
    repo_root.mkdir(parents=True, exist_ok=True)

    prompt_file = repo_root / "HeadlessPrompt.txt"
    prompt_file.write_text("  hola headless  \n", encoding="utf-8")

    prompt = launcher._read_headless_prompt(repo_root)
    assert prompt == "hola headless"

    output_path = launcher._write_headless_result(repo_root, "respuesta")
    assert output_path.is_file()
    assert output_path.parent == repo_root / "Results"
    assert output_path.read_text(encoding="utf-8") == "respuesta\n"
    latest = repo_root / "Results" / "latest.txt"
    assert latest.read_text(encoding="utf-8") == "respuesta\n"


def test_main_headless_uses_default_profile(monkeypatch, tmp_path):
    launcher = _load_launcher_module()

    class _DummyProfileManager:
        def __init__(self, repo_root):
            self.repo_root = repo_root
            self.default_selected = False

        def set_active_profile(self, name):
            if name != "default":
                raise AssertionError(f"Expected default profile, got {name}")
            self.default_selected = True
            return SimpleNamespace(
                name="default",
                runtime_ui="tui",
                runtime_provider="openai",
                runtime_model="gpt-4o-mini",
            )

        def get_active_profile(self):
            raise AssertionError("get_active_profile should not be used in headless mode")

    class _DummyRegistry:
        def __init__(self, repo_root):
            del repo_root

        def names(self):
            return ["tui"]

    captured: dict[str, str] = {}

    def _fake_run_headless(repo_root, *, provider, model, prompt):
        captured["provider"] = provider
        captured["model"] = model
        captured["prompt"] = prompt
        assert repo_root == launcher.REPO_ROOT
        return 0

    monkeypatch.setattr(launcher, "AgentProfileManager", _DummyProfileManager)
    monkeypatch.setattr(launcher, "UIRegistry", _DummyRegistry)
    monkeypatch.setattr(launcher, "_discover_providers", lambda root: ["openai"])
    monkeypatch.setattr(launcher, "_read_headless_prompt", lambda root: "prompt desde archivo")
    monkeypatch.setattr(launcher, "_run_headless", _fake_run_headless)
    monkeypatch.setattr("sys.argv", ["AIDZero.py", "--headless", "--agent", "otro"])

    exit_code = launcher.main()
    assert exit_code == 0
    assert captured == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "prompt": "prompt desde archivo",
    }


def test_main_thirdparty_ui_runs_core_only(monkeypatch):
    launcher = _load_launcher_module()
    captured: dict[str, object] = {}

    class _DummyProfileManager:
        def __init__(self, repo_root):
            del repo_root

        def get_active_profile(self):
            return SimpleNamespace(
                name="default",
                runtime_ui="AndroidApp",
                runtime_provider="lmstudio",
                runtime_model="zai-org/glm-4.7-flash",
            )

        def set_active_profile(self, name):
            del name
            return self.get_active_profile()

    class _DummyRegistry:
        def __init__(self, repo_root):
            del repo_root

        def names(self):
            return ["AndroidApp"]

        def ui_type(self, ui_name):
            del ui_name
            return "thirdparty"

    def _fake_serve_core_api(*, repo_root, provider_name, model, host, port):
        captured["repo_root"] = repo_root
        captured["provider_name"] = provider_name
        captured["model"] = model
        captured["host"] = host
        captured["port"] = port
        return 0

    monkeypatch.setattr(launcher, "AgentProfileManager", _DummyProfileManager)
    monkeypatch.setattr(launcher, "UIRegistry", _DummyRegistry)
    monkeypatch.setattr(launcher, "_discover_providers", lambda root: ["lmstudio"])
    monkeypatch.setattr(launcher, "serve_core_api", _fake_serve_core_api)
    monkeypatch.setattr("sys.argv", ["AIDZero.py"])

    exit_code = launcher.main()

    assert exit_code == 0
    assert captured == {
        "repo_root": launcher.REPO_ROOT,
        "provider_name": "lmstudio",
        "model": "zai-org/glm-4.7-flash",
        "host": "0.0.0.0",
        "port": 8765,
    }
