from __future__ import annotations

from CORE.agents import AgentProfileManager
from CORE.memory import MemoryStore
from CORE.tooling import build_default_tool_registry


def _write_profile(agents_dir, name: str, json_text: str, *, prompt_text: str = "Base prompt\n"):
    profile_dir = agents_dir / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "system_prompt.md").write_text(prompt_text, encoding="utf-8")
    (profile_dir / f"{name}.json").write_text(json_text, encoding="utf-8")
    return profile_dir


def test_agent_profile_manager_reads_active_profile(tmp_path):
    agents_dir = tmp_path / "Agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    _write_profile(
        agents_dir,
        "default",
        """
        {
          "name": "default",
          "system_prompt_file": "system_prompt.md",
          "runtime": {"ui": "terminal", "provider": "openai", "model": "gpt-4o-mini"},
          "modules": {"tools": "all", "dash": "all"}
        }
        """,
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
    _write_profile(
        agents_dir,
        "default",
        """
        {
          "name": "default",
          "system_prompt_file": "system_prompt.md",
          "runtime": {"ui": "terminal", "provider": "openai", "model": "gpt-4o-mini"},
          "features": {"memory": false, "history": false},
          "modules": {"tools": "all", "dash": "all"}
        }
        """,
    )

    manager = AgentProfileManager(tmp_path)
    profile = manager.get_active_profile()

    assert profile.memory_enabled is False
    assert profile.history_enabled is False


def test_agent_profile_manager_reads_agents_from_repo_parent_when_repo_is_code_root(tmp_path):
    repo_root = tmp_path / "AIDZeroCode"
    repo_root.mkdir(parents=True, exist_ok=True)
    agents_dir = tmp_path / "Agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    default_dir = _write_profile(
        agents_dir,
        "default",
        """
        {
          "name": "default",
          "system_prompt_file": "system_prompt.md",
          "runtime": {"ui": "terminal", "provider": "openai", "model": "gpt-4o-mini"}
        }
        """,
    )

    manager = AgentProfileManager(repo_root)
    profile = manager.get_active_profile()

    assert profile.name == "default"
    assert profile.system_prompt == "Base prompt"
    assert profile.source_path == default_dir / "default.json"


def test_agent_profile_manager_rejects_invalid_feature_flag(tmp_path):
    agents_dir = tmp_path / "Agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    _write_profile(
        agents_dir,
        "default",
        """
        {
          "name": "default",
          "system_prompt_file": "system_prompt.md",
          "runtime": {"ui": "terminal", "provider": "openai", "model": "gpt-4o-mini"},
          "features": {"memory": "yes"}
        }
        """,
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

    _write_profile(
        agents_dir,
        "default",
        """
        {
          "name": "default",
          "system_prompt_file": "../../core/system_prompt.md",
          "runtime": {"ui": "terminal", "provider": "openai", "model": "gpt-4o-mini"},
          "modules": {"tools": "all", "dash": "all"}
        }
        """,
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
    _write_profile(
        agents_dir,
        "default",
        """
        {
          "name": "default",
          "system_prompt_file": "system_prompt.md"
        }
        """,
    )

    manager = AgentProfileManager(tmp_path)
    try:
        manager.get_active_profile()
        assert False, "Expected RuntimeError for missing runtime config"
    except RuntimeError as error:
        assert "runtime" in str(error)


def test_agent_profile_manager_reads_headless_prompt_from_profile_folder(tmp_path):
    agents_dir = tmp_path / "Agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = _write_profile(
        agents_dir,
        "default",
        """
        {
          "name": "default",
          "system_prompt_file": "system_prompt.md",
          "runtime": {"ui": "terminal", "provider": "openai", "model": "gpt-4o-mini"}
        }
        """,
    )
    (profile_dir / "HeadlessPrompt.txt").write_text("  hola headless  \n", encoding="utf-8")

    manager = AgentProfileManager(tmp_path)
    prompt = manager.get_headless_prompt("default")

    assert prompt == "hola headless"


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
