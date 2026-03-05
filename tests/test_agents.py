from __future__ import annotations

from core.agents import AgentProfileManager
from core.memory import MemoryStore
from core.tooling import build_default_tool_registry


def test_agent_profile_manager_reads_active_profile(tmp_path):
    agents_dir = tmp_path / "Agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "system_prompt.md").write_text("Base prompt\n", encoding="utf-8")

    (agents_dir / "default.json").write_text(
        """
        {
          "name": "default",
          "system_prompt_file": "system_prompt.md",
          "runtime": {"ui": "terminal", "provider": "openai", "model": "gpt-4o-mini"},
          "modules": {"tools": "all", "dash": "all"}
        }
        """,
        encoding="utf-8",
    )

    manager = AgentProfileManager(tmp_path)
    profile = manager.get_active_profile()

    assert profile.name == "default"
    assert profile.system_prompt == "Base prompt"
    assert profile.enabled_tools is None
    assert profile.enabled_dash_modules is None
    assert profile.memory_enabled is True
    assert profile.history_enabled is True
    assert profile.runtime_ui == "terminal"
    assert profile.runtime_provider == "openai"
    assert profile.runtime_model == "gpt-4o-mini"


def test_agent_profile_manager_parses_memory_and_history_flags(tmp_path):
    agents_dir = tmp_path / "Agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "system_prompt.md").write_text("Base prompt\n", encoding="utf-8")
    (agents_dir / "default.json").write_text(
        """
        {
          "name": "default",
          "system_prompt_file": "system_prompt.md",
          "runtime": {"ui": "terminal", "provider": "openai", "model": "gpt-4o-mini"},
          "features": {"memory": false, "history": false},
          "modules": {"tools": "all", "dash": "all"}
        }
        """,
        encoding="utf-8",
    )

    manager = AgentProfileManager(tmp_path)
    profile = manager.get_active_profile()

    assert profile.memory_enabled is False
    assert profile.history_enabled is False


def test_agent_profile_manager_rejects_invalid_feature_flag(tmp_path):
    agents_dir = tmp_path / "Agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "system_prompt.md").write_text("Base prompt\n", encoding="utf-8")
    (agents_dir / "default.json").write_text(
        """
        {
          "name": "default",
          "system_prompt_file": "system_prompt.md",
          "runtime": {"ui": "terminal", "provider": "openai", "model": "gpt-4o-mini"},
          "features": {"memory": "yes"}
        }
        """,
        encoding="utf-8",
    )

    manager = AgentProfileManager(tmp_path)
    try:
        manager.get_active_profile()
        assert False, "Expected RuntimeError for invalid features.memory type"
    except RuntimeError as error:
        assert "features.memory" in str(error)


def test_agent_profile_manager_rejects_prompt_file_outside_agents(tmp_path):
    agents_dir = tmp_path / "Agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "core").mkdir(parents=True, exist_ok=True)
    (tmp_path / "core" / "system_prompt.md").write_text("Outside prompt\n", encoding="utf-8")

    (agents_dir / "default.json").write_text(
        """
        {
          "name": "default",
          "system_prompt_file": "../core/system_prompt.md",
          "runtime": {"ui": "terminal", "provider": "openai", "model": "gpt-4o-mini"},
          "modules": {"tools": "all", "dash": "all"}
        }
        """,
        encoding="utf-8",
    )

    manager = AgentProfileManager(tmp_path)
    try:
        manager.get_active_profile()
        assert False, "Expected RuntimeError for prompt path outside Agents/"
    except RuntimeError as error:
        assert "within Agents/" in str(error)


def test_agent_profile_manager_rejects_missing_runtime_config(tmp_path):
    agents_dir = tmp_path / "Agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "system_prompt.md").write_text("Base prompt\n", encoding="utf-8")
    (agents_dir / "default.json").write_text(
        """
        {
          "name": "default",
          "system_prompt_file": "system_prompt.md"
        }
        """,
        encoding="utf-8",
    )

    manager = AgentProfileManager(tmp_path)
    try:
        manager.get_active_profile()
        assert False, "Expected RuntimeError for missing runtime config"
    except RuntimeError as error:
        assert "runtime" in str(error)


def test_build_default_tool_registry_can_filter_tools_by_agent_profile(tmp_path):
    tools_dir = tmp_path / "TOOLS"
    tools_dir.mkdir(parents=True, exist_ok=True)

    (tools_dir / "alpha.py").write_text(
        "\n".join(
            [
                'TOOL_NAME = "alpha"',
                'TOOL_DESCRIPTION = "Alpha"',
                'TOOL_PARAMETERS = {"type": "object"}',
                "",
                "def run(arguments, *, repo_root, memory):",
                "    del arguments, repo_root, memory",
                "    return {'ok': True}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tools_dir / "beta.py").write_text(
        "\n".join(
            [
                'TOOL_NAME = "beta"',
                'TOOL_DESCRIPTION = "Beta"',
                'TOOL_PARAMETERS = {"type": "object"}',
                "",
                "def run(arguments, *, repo_root, memory):",
                "    del arguments, repo_root, memory",
                "    return {'ok': True}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    memory = MemoryStore(tmp_path / ".aidzero" / "memory.json")
    registry = build_default_tool_registry(tmp_path, memory, enabled_names=["beta"])

    assert registry.names() == ["beta"]


def test_build_default_tool_registry_can_exclude_tools(tmp_path):
    tools_dir = tmp_path / "TOOLS"
    tools_dir.mkdir(parents=True, exist_ok=True)

    (tools_dir / "alpha.py").write_text(
        "\n".join(
            [
                'TOOL_NAME = "alpha"',
                'TOOL_DESCRIPTION = "Alpha"',
                'TOOL_PARAMETERS = {"type": "object"}',
                "",
                "def run(arguments, *, repo_root, memory):",
                "    del arguments, repo_root, memory",
                "    return {'ok': True}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tools_dir / "beta.py").write_text(
        "\n".join(
            [
                'TOOL_NAME = "beta"',
                'TOOL_DESCRIPTION = "Beta"',
                'TOOL_PARAMETERS = {"type": "object"}',
                "",
                "def run(arguments, *, repo_root, memory):",
                "    del arguments, repo_root, memory",
                "    return {'ok': True}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    memory = MemoryStore(tmp_path / ".aidzero" / "memory.json")
    registry = build_default_tool_registry(tmp_path, memory, disabled_names=["alpha"])

    assert registry.names() == ["beta"]
