from __future__ import annotations

import json

from DASH import setup


class _FakeApp:
    def __init__(self, repo_root) -> None:
        self.repo_root = repo_root
        self.lines: list[str] = []

    def _append_system_line(self, text: str) -> None:
        self.lines.append(text)


def test_setup_cron_generates_cron_assets(tmp_path):
    app = _FakeApp(tmp_path)
    (tmp_path / ".aidzero").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".aidzero" / "runtime_config.json").write_text(
        json.dumps({"ui": "terminal", "provider": "openai", "model": "gpt-4o-mini"}) + "\n",
        encoding="utf-8",
    )

    handled = setup.run("/setup cron */5 * * * *", app=app)

    assert handled is True
    config = json.loads((tmp_path / ".aidzero" / "trigger_sources.json").read_text(encoding="utf-8"))
    assert config["cron_schedule"] == "*/5 * * * *"
    assert (tmp_path / ".aidzero" / "scripts" / "run_cron.sh").exists()
    assert (tmp_path / ".aidzero" / "setup" / "aidzero.crontab").exists()
    assert any("Generated .aidzero/setup/aidzero.crontab" in line for line in app.lines)


def test_setup_registers_message_origin(tmp_path):
    app = _FakeApp(tmp_path)

    handled = setup.run("/setup message-origin zendesk .aidzero/inbox/zendesk.jsonl", app=app)

    assert handled is True
    config = json.loads((tmp_path / ".aidzero" / "trigger_sources.json").read_text(encoding="utf-8"))
    assert config["message_origins"] == [{"name": "zendesk", "path": ".aidzero/inbox/zendesk.jsonl"}]
    assert (tmp_path / ".aidzero" / "inbox" / "zendesk.jsonl").exists()


def test_setup_registers_webhook_origin_with_default_path(tmp_path):
    app = _FakeApp(tmp_path)

    handled = setup.run("/setup webhook-origin stripe", app=app)

    assert handled is True
    config = json.loads((tmp_path / ".aidzero" / "trigger_sources.json").read_text(encoding="utf-8"))
    assert config["webhook_origins"] == [{"name": "stripe", "path": ".aidzero/inbox/webhooks_stripe.jsonl"}]
    assert (tmp_path / ".aidzero" / "inbox" / "webhooks_stripe.jsonl").exists()
