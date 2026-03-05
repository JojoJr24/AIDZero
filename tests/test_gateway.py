from __future__ import annotations

from core.gateway import TriggerGateway


def test_gateway_collects_only_interactive_prompt(tmp_path):
    gateway = TriggerGateway(tmp_path)
    events = gateway.collect(trigger="interactive", prompt="interactive task")
    assert len(events) == 1
    assert events[0].kind == "interactive"
    assert events[0].prompt == "interactive task"


def test_gateway_ignores_non_interactive_trigger(tmp_path):
    gateway = TriggerGateway(tmp_path)
    events = gateway.collect(trigger="all", prompt="interactive task")
    assert events == []
