"""Shared defaults for generated agent runtime config."""

from __future__ import annotations

from typing import Any

from .models import AgentPlan

AGENT_CONFIG_FILENAME = "agent_config.json"


def build_default_runtime_config(plan: AgentPlan) -> dict[str, Any]:
    """Build a default runtime config for a generated agent."""
    selected_provider = plan.required_llm_providers[0] if plan.required_llm_providers else ""
    return {
        "provider": selected_provider,
        "model": "",
        "provider_options": {},
        "generation_config": {
            "temperature": 0.2,
        },
    }
