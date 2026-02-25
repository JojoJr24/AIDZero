"""Tests for repository-clone scaffolding behavior."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.generated_agent_config import AGENT_CONFIG_FILENAME
from agent.models import AgentPlan, ComponentCatalog
from agent.scaffold import AgentScaffolder


def _seed_runtime_support(source_root: Path) -> None:
    runtime_support_dir = source_root / "agent"
    runtime_support_dir.mkdir(parents=True, exist_ok=True)
    (runtime_support_dir / "provider_base.py").write_text(
        "class LLMProvider: pass\n",
        encoding="utf-8",
    )
    (runtime_support_dir / "openai_compatible_provider.py").write_text(
        "class OpenAICompatibleProvider: pass\n",
        encoding="utf-8",
    )
    (runtime_support_dir / "generated_agent_runtime.py").write_text(
        "def create_provider_from_config(*, project_root, config):\n    return object()\n",
        encoding="utf-8",
    )
    (runtime_support_dir / "workspace_guard.py").write_text(
        "class WorkspaceGuard:\n    def __init__(self, workspace_root):\n        self.workspace_root = workspace_root\n",
        encoding="utf-8",
    )
    (runtime_support_dir / "private_logic.py").write_text("SHOULD_NOT_COPY = True\n", encoding="utf-8")


def test_scaffold_clones_repo_and_rebuilds_agent_runtime_package(tmp_path: Path) -> None:
    source_root = tmp_path / "source_repo"
    source_root.mkdir(parents=True)

    (source_root / "AIDZero.py").write_text("print('root runtime')\n", encoding="utf-8")
    (source_root / "README.md").write_text("# demo\n", encoding="utf-8")
    (source_root / "main.py").write_text("print('source')\n", encoding="utf-8")
    (source_root / "LICENSE").write_text("private root file\n", encoding="utf-8")
    (source_root / "generated_agents" / "old").mkdir(parents=True)
    (source_root / "generated_agents" / "old" / "artifact.txt").write_text("old\n", encoding="utf-8")
    (source_root / "MCP").mkdir(parents=True)
    (source_root / "MCP" / "mcporter.json").write_text('{"mcpServers":{}}\n', encoding="utf-8")

    provider_dir = source_root / "LLMProviders" / "AID-openai"
    provider_dir.mkdir(parents=True)
    (provider_dir / "provider.py").write_text(
        (
            "from agent.openai_compatible_provider import OpenAICompatibleProvider\n\n"
            "class OpenAIProvider(OpenAICompatibleProvider):\n"
            "    pass\n"
        ),
        encoding="utf-8",
    )
    (source_root / "LLMProviders" / "base.py").write_text("class LLMProvider:\n    pass\n", encoding="utf-8")

    _seed_runtime_support(source_root)

    plan = AgentPlan(
        agent_name="demo",
        project_folder="generated_agents/demo",
        goal="goal",
        summary="summary",
        required_llm_providers=["AID-openai"],
    )
    catalog = ComponentCatalog(root=source_root)
    destination = source_root / "generated_agents" / "demo"

    result = AgentScaffolder(source_root).scaffold(
        destination=destination,
        plan=plan,
        catalog=catalog,
        user_request="build reporting workflows",
        main_py_source="print('generated')\n",
    )

    runtime_config_path = destination / AGENT_CONFIG_FILENAME
    assert result.runtime_config_file == runtime_config_path
    assert runtime_config_path.exists()
    with runtime_config_path.open("r", encoding="utf-8") as handle:
        config_payload = json.load(handle)
    assert config_payload["provider"] == "AID-openai"

    assert (destination / "LLMProviders").exists()
    assert (destination / "MCP" / "mcporter.json").exists()
    assert not (destination / "AIDZero.py").exists()
    assert not (destination / "README.md").exists()
    assert not (destination / "LICENSE").exists()
    assert (destination / "LLMProviders" / "AID-openai" / "provider.py").exists()

    assert (destination / "agent" / "__init__.py").exists()
    assert (destination / "agent" / "provider_base.py").exists()
    assert (destination / "agent" / "openai_compatible_provider.py").exists()
    assert (destination / "agent" / "generated_agent_runtime.py").exists()
    assert not (destination / "agent" / "workspace_guard.py").exists()
    assert (destination / "agent" / "child_manifest.json").exists()
    assert not (destination / "agent" / "private_logic.py").exists()

    assert not (destination / "generated_agents").exists()
    assert (destination / "main.py").read_text(encoding="utf-8") == "print('generated')\n"
    with (destination / "agent" / "child_manifest.json").open("r", encoding="utf-8") as handle:
        child_manifest = json.load(handle)
    assert child_manifest["original_user_request"] == "build reporting workflows"
    assert child_manifest["default_task"] == "build reporting workflows"
    assert child_manifest["agent_name"] == "demo"
    assert child_manifest["goal"] == "goal"

    module_file = destination / "LLMProviders" / "AID-openai" / "provider.py"
    module_spec = importlib.util.spec_from_file_location("generated_provider", module_file)
    assert module_spec is not None
    assert module_spec.loader is not None
    sys.path.insert(0, str(destination))
    try:
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        assert hasattr(module, "OpenAIProvider")
    finally:
        sys.path.pop(0)


def test_scaffold_rejects_non_empty_destination_without_overwrite(tmp_path: Path) -> None:
    source_root = tmp_path / "source_repo"
    source_root.mkdir(parents=True)
    (source_root / "README.md").write_text("# source\n", encoding="utf-8")
    (source_root / "LLMProviders" / "AID-openai").mkdir(parents=True)
    (source_root / "LLMProviders" / "AID-openai" / "provider.py").write_text("class OpenAIProvider: pass\n", encoding="utf-8")
    _seed_runtime_support(source_root)

    destination = source_root / "generated"
    destination.mkdir(parents=True)
    (destination / "exists.txt").write_text("busy\n", encoding="utf-8")

    plan = AgentPlan(
        agent_name="demo",
        project_folder="generated/demo",
        goal="goal",
        summary="summary",
        required_llm_providers=["AID-openai"],
    )

    with pytest.raises(FileExistsError):
        AgentScaffolder(source_root).scaffold(
            destination=destination,
            plan=plan,
            catalog=ComponentCatalog(root=source_root),
            user_request="build workflow",
            main_py_source="print('generated')\n",
            overwrite=False,
        )
