from __future__ import annotations

import json

from agent.gateway import TriggerGateway


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
