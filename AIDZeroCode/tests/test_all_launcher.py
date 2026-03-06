from __future__ import annotations

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
