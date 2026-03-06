from __future__ import annotations

from types import SimpleNamespace

from CORE import all_launcher


def test_all_launcher_defaults_to_lan_bind(monkeypatch):
    monkeypatch.setattr("sys.argv", ["aidzero-all"])

    args = all_launcher._parse_args()

    assert args.host == "0.0.0.0"
    assert args.port == 8765


def test_connect_host_uses_loopback_for_wildcard_bind():
    assert all_launcher._connect_host("0.0.0.0") == "127.0.0.1"
    assert all_launcher._connect_host("::") == "127.0.0.1"
    assert all_launcher._connect_host("[::]") == "127.0.0.1"


def test_connect_host_preserves_specific_host():
    assert all_launcher._connect_host("192.168.1.20") == "192.168.1.20"


def test_all_launcher_thirdparty_runs_only_core(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

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
        calls["repo_root"] = repo_root
        calls["provider_name"] = provider_name
        calls["model"] = model
        calls["host"] = host
        calls["port"] = port
        return 0

    monkeypatch.setattr(all_launcher, "AgentProfileManager", _DummyProfileManager)
    monkeypatch.setattr(all_launcher, "UIRegistry", _DummyRegistry)
    monkeypatch.setattr(all_launcher, "serve_core_api", _fake_serve_core_api)
    monkeypatch.setattr(
        "sys.argv",
        [
            "aidzero-all",
            "--repo-root",
            str(tmp_path),
            "--host",
            "0.0.0.0",
            "--port",
            "8765",
            "--request",
            "hola",
        ],
    )

    exit_code = all_launcher.main()

    assert exit_code == 0
    assert calls == {
        "repo_root": tmp_path.resolve(),
        "provider_name": "lmstudio",
        "model": "zai-org/glm-4.7-flash",
        "host": "0.0.0.0",
        "port": 8765,
    }
