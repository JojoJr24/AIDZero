from __future__ import annotations

import json
from pathlib import Path

from agent.models import AgentPlan, ComponentCatalog, ComponentItem
from agent.service import _build_plan_from_response


def _catalog() -> ComponentCatalog:
    return ComponentCatalog(
        root=Path.cwd(),
        llm_providers=[
            ComponentItem(name="AID-openai", path=Path("LLMProviders/AID-openai")),
            ComponentItem(name="AID-google_gemini", path=Path("LLMProviders/AID-google_gemini")),
        ],
        skills=[
            ComponentItem(name="AID-skill-creator", path=Path("SKILLS/AID-skill-creator")),
            ComponentItem(name="AID-other", path=Path("SKILLS/AID-other")),
        ],
        tools=[
            ComponentItem(name="AID-skill-tool", path=Path("TOOLS/AID-skill-tool")),
            ComponentItem(name="AID-extra-tool", path=Path("TOOLS/AID-extra-tool")),
        ],
        mcp=[
            ComponentItem(name="AID-tool-gateway", path=Path("MCP/AID-tool-gateway")),
            ComponentItem(name="AID-mcp-other", path=Path("MCP/AID-mcp-other")),
        ],
        ui=[
            ComponentItem(name="terminal", path=Path("UI/terminal")),
            ComponentItem(name="web", path=Path("UI/web")),
        ],
    )


def _base_plan() -> AgentPlan:
    return AgentPlan(
        agent_name="BasePlan",
        project_folder="agent_base_plan",
        goal="Base goal",
        summary="Base summary",
        required_llm_providers=["AID-openai"],
        required_skills=["AID-skill-creator"],
        required_tools=["AID-skill-tool"],
        required_mcp=["AID-tool-gateway"],
        required_ui=["terminal"],
        folder_blueprint=["agent/", "LLMProviders/"],
        implementation_steps=["s1", "s2", "s3"],
        warnings=["base warning"],
        raw_response="base",
    )


def test_build_plan_from_json_response_parses_and_normalizes_components() -> None:
    response = json.dumps(
        {
            "agent_name": "DailyReporter",
            "project_folder": "Daily Reporter Agent",
            "goal": "Generate KPI reports daily.",
            "summary": "Builds and sends daily reports.",
            "required_llm_providers": ["openai"],
            "required_skills": ["AID-skill-creator"],
            "required_tools": ["skill-tool"],
            "required_mcp": ["tool-gateway"],
            "required_ui": ["terminal"],
            "folder_blueprint": ["agent/", "UI/"],
            "implementation_steps": ["step1", "step2", "step3"],
            "warnings": ["review cron schedule"],
        }
    )

    plan = _build_plan_from_response(
        user_request="Build a daily reporting agent.",
        response_text=response,
        catalog=_catalog(),
    )

    assert plan.project_folder == "agent_daily_reporter_agent"
    assert plan.agent_name == "DailyReporter"
    assert plan.required_llm_providers == ["AID-openai"]
    assert plan.required_skills == ["AID-skill-creator"]
    assert plan.required_tools == ["AID-skill-tool"]
    assert plan.required_mcp == ["AID-tool-gateway"]
    assert plan.required_ui == ["terminal"]
    assert plan.folder_blueprint == ["agent/", "UI/"]
    assert plan.implementation_steps == ["step1", "step2", "step3"]
    assert plan.warnings == ["review cron schedule"]


def test_build_plan_falls_back_to_base_plan_when_response_is_not_json() -> None:
    base_plan = _base_plan()
    response = "Please keep terminal only and improve error handling."

    plan = _build_plan_from_response(
        user_request="Build a daily reporting agent.",
        response_text=response,
        catalog=_catalog(),
        base_plan=base_plan,
    )

    assert plan.project_folder == base_plan.project_folder
    assert plan.required_llm_providers == base_plan.required_llm_providers
    assert plan.required_skills == base_plan.required_skills
    assert plan.required_tools == base_plan.required_tools
    assert plan.required_mcp == base_plan.required_mcp
    assert plan.required_ui == base_plan.required_ui
    assert plan.warnings == base_plan.warnings
    assert plan.summary.startswith("Please keep terminal only")


def test_build_plan_respects_explicit_empty_component_lists() -> None:
    base_plan = _base_plan()
    response = json.dumps(
        {
            "required_llm_providers": ["AID-openai"],
            "required_skills": [],
            "required_tools": [],
            "required_mcp": [],
            "required_ui": ["terminal"],
        }
    )

    plan = _build_plan_from_response(
        user_request="Build a daily reporting agent.",
        response_text=response,
        catalog=_catalog(),
        base_plan=base_plan,
    )

    assert plan.required_llm_providers == ["AID-openai"]
    assert plan.required_skills == []
    assert plan.required_tools == []
    assert plan.required_mcp == []
    assert plan.required_ui == ["terminal"]
