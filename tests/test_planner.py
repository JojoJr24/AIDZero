"""Tests for provider selection normalization in planning."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.models import ComponentCatalog, ComponentItem
from agent.planner import AgentPlanner


def _catalog_with_providers(provider_names: list[str]) -> ComponentCatalog:
    providers = [
        ComponentItem(
            name=name,
            path=Path(f"LLMProviders/{name}"),
            kind="llm_provider",
        )
        for name in provider_names
    ]
    return ComponentCatalog(root=REPO_ROOT, llm_providers=providers)


def test_validate_defaults_provider_when_missing() -> None:
    planner = AgentPlanner(provider=object(), model="test-model")
    catalog = _catalog_with_providers(["AID-openai", "AID-google_gemini"])

    plan = planner._validate_against_catalog(  # noqa: SLF001
        {
            "agent_name": "demo",
            "project_folder": "generated_agents/demo",
            "goal": "goal",
            "summary": "summary",
            "required_llm_providers": [],
        },
        catalog,
    )

    assert plan.required_llm_providers == ["AID-openai"]
    assert any("missing required_llm_providers" in warning for warning in plan.warnings)


def test_validate_uses_single_provider_even_when_multiple_selected() -> None:
    planner = AgentPlanner(provider=object(), model="test-model")
    catalog = _catalog_with_providers(["AID-openai", "AID-google_gemini"])

    plan = planner._validate_against_catalog(  # noqa: SLF001
        {
            "agent_name": "demo",
            "project_folder": "generated_agents/demo",
            "goal": "goal",
            "summary": "summary",
            "required_llm_providers": ["AID-google_gemini", "AID-openai"],
        },
        catalog,
    )

    assert plan.required_llm_providers == ["AID-google_gemini"]
    assert any("multiple providers" in warning for warning in plan.warnings)
