from __future__ import annotations

from pathlib import Path

from core.agents import AgentProfile
from core.api_contract import profile_from_dict, profile_to_dict


def test_profile_contract_roundtrip_includes_runtime_fields(tmp_path):
    profile = AgentProfile(
        name="default",
        description="Default profile",
        system_prompt="Be concise",
        enabled_tools=["sandbox_run"],
        enabled_dash_modules=None,
        memory_enabled=True,
        history_enabled=False,
        runtime_ui="tui",
        runtime_provider="openai",
        runtime_model="gpt-4o-mini",
        source_path=tmp_path / "Agents" / "default.json",
    )

    payload = profile_to_dict(profile)
    rebuilt = profile_from_dict(payload, repo_root=Path(tmp_path))

    assert rebuilt.runtime_ui == "tui"
    assert rebuilt.runtime_provider == "openai"
    assert rebuilt.runtime_model == "gpt-4o-mini"
    assert rebuilt.history_enabled is False
