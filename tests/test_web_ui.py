"""Tests for the AIDZero web UI helpers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UI.web.agent_web import _build_options_payload, _coerce_bool, _model_candidates, _plan_from_payload


class _RegistryStub:
    def __init__(self) -> None:
        self._providers = ["AID-alpha", "AID-beta"]

    def names(self) -> list[str]:
        return list(self._providers)

    def has(self, provider_name: str) -> bool:
        return provider_name in self._providers

    def default_model(self, provider_name: str) -> str:
        if provider_name == "AID-alpha":
            return "alpha-default"
        if provider_name == "AID-beta":
            return "beta-default"
        raise ValueError("Unknown provider")

    def try_list_models(self, provider_name: str) -> list[str]:
        if provider_name == "AID-alpha":
            return ["alpha-fast", "alpha-fast", "alpha-pro"]
        if provider_name == "AID-beta":
            raise RuntimeError("listing not available")
        return []


def test_coerce_bool_variants() -> None:
    assert _coerce_bool(True) is True
    assert _coerce_bool(False) is False
    assert _coerce_bool("true") is True
    assert _coerce_bool(" YES ") is True
    assert _coerce_bool("0") is False
    assert _coerce_bool("off") is False
    assert _coerce_bool("unknown", default=True) is True
    assert _coerce_bool(None, default=False) is False


def test_model_candidates_dedup_and_default_insert() -> None:
    registry = _RegistryStub()
    models, model_error = _model_candidates(registry, "AID-alpha")
    assert model_error is None
    assert models == ["alpha-default", "alpha-fast", "alpha-pro"]


def test_build_options_payload_fallbacks_to_first_provider() -> None:
    registry = _RegistryStub()
    payload = _build_options_payload(
        provider_registry=registry,
        default_provider="AID-missing",
        default_model="missing-model",
        default_request="plan a testing agent",
        dry_run=True,
        overwrite=False,
        prompt_history=["first prompt", "second prompt"],
    )

    assert payload["selected_provider"] == "AID-alpha"
    assert payload["selected_model"] == "alpha-default"
    assert payload["providers"] == [
        {"id": "AID-alpha", "label": "alpha"},
        {"id": "AID-beta", "label": "beta"},
    ]
    assert payload["models"] == [
        {"id": "alpha-default", "label": "alpha-default"},
        {"id": "alpha-fast", "label": "alpha-fast"},
        {"id": "alpha-pro", "label": "alpha-pro"},
    ]
    assert payload["default_request"] == "plan a testing agent"
    assert payload["dry_run"] is True
    assert payload["overwrite"] is False
    assert payload["prompt_history"] == ["first prompt", "second prompt"]


def test_build_options_payload_uses_default_model_when_listing_fails() -> None:
    registry = _RegistryStub()
    payload = _build_options_payload(
        provider_registry=registry,
        default_provider="AID-beta",
        default_model="beta-default",
        default_request="",
        dry_run=False,
        overwrite=True,
        prompt_history=[],
    )

    assert payload["models"] == [{"id": "beta-default", "label": "beta-default"}]
    assert payload["selected_model"] == "beta-default"
    assert payload["model_error"] == "listing not available"


def test_plan_from_payload_builds_agent_plan() -> None:
    plan = _plan_from_payload(
        {
            "agent_name": "demo",
            "project_folder": "generated_agents/demo",
            "goal": "goal",
            "summary": "summary",
            "required_llm_providers": ["AID-alpha"],
            "required_skills": ["AID-skill"],
            "required_tools": ["AID-tool"],
            "required_mcp": ["AID-mcp"],
            "required_ui": ["terminal"],
            "folder_blueprint": ["src", "tests"],
            "implementation_steps": ["step one"],
            "warnings": ["warn"],
            "raw_response": "raw",
        }
    )
    assert plan.agent_name == "demo"
    assert plan.project_folder == "generated_agents/demo"
    assert plan.required_llm_providers == ["AID-alpha"]
    assert plan.required_ui == ["terminal"]


def test_plan_from_payload_requires_required_fields() -> None:
    try:
        _plan_from_payload({})
    except ValueError as error:
        assert "agent_name" in str(error)
    else:
        raise AssertionError("Expected ValueError for missing required plan fields")
