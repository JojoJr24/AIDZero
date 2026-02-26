from __future__ import annotations

from pathlib import Path

from agent.models import ComponentCatalog
from agent.provider_base import ProviderBaseRuntime


class _DummyProvider:
    pass


def _catalog() -> ComponentCatalog:
    return ComponentCatalog(
        root=Path.cwd(),
        llm_providers=[],
        skills=[],
        tools=[],
        mcp=[],
        ui=[],
    )


def test_system_prompt_enforces_child_scaffold_rules() -> None:
    runtime = ProviderBaseRuntime(
        provider=_DummyProvider(),
        model="fake-model",
        repo_root=Path.cwd(),
        catalog=_catalog(),
    )

    prompt = runtime._build_system_prompt(ui_name="terminal")

    assert "Critical scope rule" in prompt
    assert "newly scaffolded child workspace" in prompt
    assert "child root (`main.py`)" in prompt
    assert "child `agent/` package" in prompt
    assert "never to the parent repository" in prompt
