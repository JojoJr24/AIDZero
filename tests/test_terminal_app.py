from __future__ import annotations

from core.gateway import TriggerGateway


def test_trigger_gateway_collects_interactive_prompt(tmp_path):
    gateway = TriggerGateway(tmp_path)

    events = gateway.collect(trigger="interactive", prompt="hola")

    assert len(events) == 1
    assert events[0].kind == "interactive"
    assert events[0].source == "terminal"
    assert events[0].prompt == "hola"


def test_trigger_gateway_ignores_empty_prompt(tmp_path):
    gateway = TriggerGateway(tmp_path)

    events = gateway.collect(trigger="interactive", prompt="  ")

    assert events == []
