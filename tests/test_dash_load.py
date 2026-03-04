from __future__ import annotations

import json

from DASH import load


class _FakeApp:
    def __init__(self, repo_root) -> None:
        self.repo_root = repo_root
        self.lines: list[str] = []

    def _append_system_line(self, text: str) -> None:
        self.lines.append(text)


def test_load_heartbeat_writes_file(tmp_path):
    app = _FakeApp(tmp_path)

    handled = load.run("/load heartbeat check status", app=app)

    assert handled is True
    assert (tmp_path / "HEARTBEAT.md").read_text(encoding="utf-8") == "check status\n"
    assert app.lines[-1] == "Updated HEARTBEAT.md."


def test_load_cron_writes_state_file(tmp_path):
    app = _FakeApp(tmp_path)

    handled = load.run("/load cron run maintenance", app=app)

    assert handled is True
    cron_path = tmp_path / ".aidzero" / "cron_prompt.txt"
    assert cron_path.read_text(encoding="utf-8") == "run maintenance\n"
    assert app.lines[-1] == "Updated .aidzero/cron_prompt.txt."


def test_load_message_appends_jsonl(tmp_path):
    app = _FakeApp(tmp_path)

    handled = load.run("/load message hola mundo", app=app)

    assert handled is True
    inbox_path = tmp_path / ".aidzero" / "inbox" / "messages.jsonl"
    rows = inbox_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0]) == {"text": "hola mundo"}


def test_load_webhook_appends_jsonl(tmp_path):
    app = _FakeApp(tmp_path)

    handled = load.run("/load webhook from integration", app=app)

    assert handled is True
    inbox_path = tmp_path / ".aidzero" / "inbox" / "webhooks.jsonl"
    rows = inbox_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0]) == {"text": "from integration"}


def test_load_returns_usage_on_invalid_command(tmp_path):
    app = _FakeApp(tmp_path)

    handled = load.run("/load heartbeat", app=app)

    assert handled is True
    assert app.lines[-1] == "Usage: /load <heartbeat|cron|message|webhook> <text>"
