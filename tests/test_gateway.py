from __future__ import annotations

import json

from core.gateway import TriggerGateway


def test_gateway_collects_all_sources(tmp_path):
    (tmp_path / "HEARTBEAT.md").write_text("heartbeat task", encoding="utf-8")
    (tmp_path / ".aidzero").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".aidzero" / "cron_prompt.txt").write_text("cron task", encoding="utf-8")

    inbox_dir = tmp_path / ".aidzero" / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    (inbox_dir / "messages.jsonl").write_text(
        json.dumps({"text": "msg task"}) + "\n",
        encoding="utf-8",
    )
    (inbox_dir / "webhooks.jsonl").write_text(
        json.dumps({"prompt": "webhook task"}) + "\n",
        encoding="utf-8",
    )

    gateway = TriggerGateway(tmp_path)
    events = gateway.collect(trigger="all", prompt="interactive task")

    kinds = [event.kind for event in events]
    assert "interactive" in kinds
    assert "heartbeat" in kinds
    assert "cron" in kinds
    assert "messages" in kinds
    assert "webhooks" in kinds

    assert not (inbox_dir / "messages.jsonl").exists()
    assert not (inbox_dir / "webhooks.jsonl").exists()


def test_gateway_uses_configured_sources(tmp_path):
    (tmp_path / ".aidzero").mkdir(parents=True, exist_ok=True)
    custom_heartbeat = tmp_path / ".aidzero" / "custom_heartbeat.txt"
    custom_heartbeat.write_text("hb custom", encoding="utf-8")
    custom_cron = tmp_path / ".aidzero" / "custom_cron.txt"
    custom_cron.write_text("cron custom", encoding="utf-8")

    ext_messages = tmp_path / "external" / "messages_a.jsonl"
    ext_messages.parent.mkdir(parents=True, exist_ok=True)
    ext_messages.write_text(json.dumps({"text": "msg custom"}) + "\n", encoding="utf-8")
    ext_webhooks = tmp_path / "external" / "webhooks_a.jsonl"
    ext_webhooks.write_text(json.dumps({"prompt": "web custom"}) + "\n", encoding="utf-8")

    config_path = tmp_path / ".aidzero" / "trigger_sources.json"
    config_path.write_text(
        json.dumps(
            {
                "heartbeat_path": ".aidzero/custom_heartbeat.txt",
                "cron_path": ".aidzero/custom_cron.txt",
                "message_origins": [{"name": "ext-a", "path": "external/messages_a.jsonl"}],
                "webhook_origins": [{"name": "ext-a", "path": "external/webhooks_a.jsonl"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    gateway = TriggerGateway(tmp_path)
    events = gateway.collect(trigger="all", prompt="interactive task")

    prompts_by_kind = {event.kind: event.prompt for event in events}
    assert prompts_by_kind["heartbeat"] == "hb custom"
    assert prompts_by_kind["cron"] == "cron custom"
    assert prompts_by_kind["messages"] == "msg custom"
    assert prompts_by_kind["webhooks"] == "web custom"
