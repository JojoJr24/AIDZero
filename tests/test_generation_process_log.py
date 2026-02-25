"""Tests for process log generation during agent scaffolding."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.models import AgentPlan, ComponentCatalog
from agent.service import AgentCreator


class _EntrypointWriterStub:
    def generate_main_py(self, *, user_request: str, plan: AgentPlan) -> str:
        del user_request, plan
        return "print('generated')\n"


class _FailingEntrypointWriterStub:
    def generate_main_py(self, *, user_request: str, plan: AgentPlan) -> str:
        del user_request, plan
        raise RuntimeError("entrypoint failed")


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


def _write_runtime_config(source_root: Path, *, log_enabled: bool) -> None:
    config_dir = source_root / ".aidzero"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "runtime_config.json").write_text(
        (
            "{\n"
            '  "ui": "terminal",\n'
            '  "provider": "AID-openai",\n'
            '  "model": "gpt-4o-mini",\n'
            f'  "generation_process_log_enabled": {str(log_enabled).lower()}\n'
            "}\n"
        ),
        encoding="utf-8",
    )


def test_generation_process_log_is_written_in_destination(tmp_path: Path) -> None:
    source_root = tmp_path / "source_repo"
    source_root.mkdir(parents=True)
    _seed_runtime_support(source_root)

    (source_root / "LLMProviders" / "AID-openai").mkdir(parents=True)
    (source_root / "LLMProviders" / "AID-openai" / "provider.py").write_text(
        "class OpenAIProvider: pass\n",
        encoding="utf-8",
    )

    plan = AgentPlan(
        agent_name="demo-logger",
        project_folder="generated_agents/demo-logger",
        goal="test goal",
        summary="test summary",
        required_llm_providers=["AID-openai"],
    )

    creator = AgentCreator(provider=object(), model="test-model", repo_root=source_root)
    creator.entrypoint_writer = _EntrypointWriterStub()

    scaffold_result = creator.create_agent_project_from_plan(
        user_request="build reporting workflow",
        plan=plan,
        catalog=ComponentCatalog(root=source_root),
        overwrite=False,
    )

    assert scaffold_result.process_log_file is not None
    assert scaffold_result.process_log_file.exists()
    assert scaffold_result.process_log_file.parent == scaffold_result.destination

    log_text = scaffold_result.process_log_file.read_text(encoding="utf-8")
    assert "generation started for agent 'demo-logger'" in log_text
    assert "starting filesystem scaffold" in log_text
    assert "copied environment folder:" in log_text
    assert "wrote child manifest:" in log_text
    assert "scaffold completed successfully" in log_text


def test_generation_process_log_written_on_failure(tmp_path: Path) -> None:
    source_root = tmp_path / "source_repo"
    source_root.mkdir(parents=True)
    _seed_runtime_support(source_root)

    plan = AgentPlan(
        agent_name="demo-failure",
        project_folder="generated_agents/demo-failure",
        goal="test goal",
        summary="test summary",
        required_llm_providers=["AID-openai"],
    )

    creator = AgentCreator(provider=object(), model="test-model", repo_root=source_root)
    creator.entrypoint_writer = _FailingEntrypointWriterStub()

    try:
        creator.create_agent_project_from_plan(
            user_request="build failing entrypoint",
            plan=plan,
            catalog=ComponentCatalog(root=source_root),
            overwrite=False,
        )
    except RuntimeError as error:
        assert "entrypoint failed" in str(error)
    else:  # pragma: no cover - should not happen
        raise AssertionError("Expected entrypoint failure to raise.")

    log_file = source_root / "generated_agents" / "demo-failure" / "generation_process.log"
    assert log_file.exists()
    log_text = log_file.read_text(encoding="utf-8")
    assert "generation started for agent 'demo-failure'" in log_text
    assert "generation failed:" in log_text


def test_generation_process_log_can_be_disabled_via_config(tmp_path: Path) -> None:
    source_root = tmp_path / "source_repo"
    source_root.mkdir(parents=True)
    _seed_runtime_support(source_root)

    (source_root / "LLMProviders" / "AID-openai").mkdir(parents=True)
    (source_root / "LLMProviders" / "AID-openai" / "provider.py").write_text(
        "class OpenAIProvider: pass\n",
        encoding="utf-8",
    )
    _write_runtime_config(source_root, log_enabled=False)

    plan = AgentPlan(
        agent_name="demo-disabled",
        project_folder="generated_agents/demo-disabled",
        goal="test goal",
        summary="test summary",
        required_llm_providers=["AID-openai"],
    )

    creator = AgentCreator(provider=object(), model="test-model", repo_root=source_root)
    creator.entrypoint_writer = _EntrypointWriterStub()

    scaffold_result = creator.create_agent_project_from_plan(
        user_request="build reporting workflow",
        plan=plan,
        catalog=ComponentCatalog(root=source_root),
        overwrite=False,
    )

    assert scaffold_result.process_log_file is None
    log_file = source_root / "generated_agents" / "demo-disabled" / "generation_process.log"
    assert not log_file.exists()
