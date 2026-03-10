from __future__ import annotations

from types import SimpleNamespace

from CORE.api_client import CoreAPIError
from CORE import ui_launcher


class _DummyProfileManager:
    def __init__(self, repo_root):
        del repo_root

    def get_active_profile(self):
        return SimpleNamespace(
            name="default",
            runtime_ui="tui",
            runtime_provider="openai",
            runtime_model="gpt-4o-mini",
        )

    def set_active_profile(self, name):
        del name
        return self.get_active_profile()


class _DummyRegistry:
    def __init__(self, repo_root):
        del repo_root

    def names(self):
        return ["tui"]

    def run(self, *args, **kwargs):
        del args, kwargs
        raise CoreAPIError("Core API request failed: <urlopen error [Errno 111] Connection refused>")

    def ui_type(self, ui_name):
        del ui_name
        return "embedded"


class _ThirdPartyRegistry(_DummyRegistry):
    def names(self):
        return ["AndroidApp"]

    def ui_type(self, ui_name):
        del ui_name
        return "thirdparty"


def test_ui_launcher_handles_core_unavailable_without_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ui_launcher, "AgentProfileManager", _DummyProfileManager)
    monkeypatch.setattr(ui_launcher, "UIRegistry", _DummyRegistry)
    monkeypatch.setattr(
        "sys.argv",
        [
            "aidzero-ui",
            "--repo-root",
            str(tmp_path),
            "--core-url",
            "http://127.0.0.1:8765",
        ],
    )

    exit_code = ui_launcher.main()

    out = capsys.readouterr().out
    assert exit_code == 2
    assert "error> cannot reach core API at http://127.0.0.1:8765" in out


def test_ui_launcher_rejects_thirdparty_ui(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ui_launcher, "AgentProfileManager", _DummyProfileManager)
    monkeypatch.setattr(ui_launcher, "UIRegistry", _ThirdPartyRegistry)
    monkeypatch.setattr(
        "sys.argv",
        [
            "aidzero-ui",
            "--repo-root",
            str(tmp_path),
            "--ui",
            "AndroidApp",
        ],
    )

    exit_code = ui_launcher.main()

    out = capsys.readouterr().out
    assert exit_code == 2
    assert "error> ui 'AndroidApp' is thirdparty and has no local launcher." in out
