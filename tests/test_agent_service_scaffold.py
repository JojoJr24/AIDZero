from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.catalog import ComponentCatalogBuilder
from agent.models import AgentPlan
from agent.service import AgentCreator


class FakeProvider:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        del model, kwargs
        self.calls.append([dict(message) for message in messages])
        return {"choices": [{"message": {"content": self.response_text}}]}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_repo(root: Path) -> None:
    _write(root / "LLMProviders/base.py", "# base\n")
    _write(root / "LLMProviders/provider_base.py", "# provider base\n")
    _write(root / "LLMProviders/openai_compatible_provider.py", "# openai compatible\n")
    _write(root / "LLMProviders/__init__.py", "")
    _write(root / "LLMProviders/AID-openai/provider.py", "class OpenAIProvider:\n    pass\n")
    _write(root / "LLMProviders/AID-google_gemini/provider.py", "class GeminiProvider:\n    pass\n")

    _write(root / "TOOLS/AID-skill-tool/tool.py", "print('skill tool')\n")
    _write(root / "TOOLS/AID-extra/tool.py", "print('extra tool')\n")
    _write(root / "SKILLS/AID-skill-creator/SKILL.md", "# Skill creator\n")
    _write(root / "SKILLS/AID-other/SKILL.md", "# Other\n")

    _write(root / "MCP/AID-tool-gateway/scripts/gateway-call.mjs", "console.log('{}');\n")
    _write(root / "MCP/mcporter.json", json.dumps({"mcpServers": {"filesystem": {"command": ["echo"]}}}))
    _write(root / "MCP/run-tool-gateway.sh", "#!/usr/bin/env bash\n")

    _write(root / "UI/AGENTS.md", "# ui\n")
    _write(root / "UI/terminal/entrypoint.py", "def run_ui(**kwargs):\n    return 0\n")
    _write(root / "UI/web/entrypoint.py", "def run_ui(**kwargs):\n    return 0\n")


def _plan(project_folder: str) -> AgentPlan:
    return AgentPlan(
        agent_name="ChildAgent",
        project_folder=project_folder,
        goal="Build a child agent that answers a request.",
        summary="summary",
        required_llm_providers=["AID-openai"],
        required_skills=["AID-skill-creator"],
        required_tools=["AID-skill-tool"],
        required_mcp=["AID-tool-gateway"],
        required_ui=["terminal"],
        folder_blueprint=["agent/", "LLMProviders/", "TOOLS/", "SKILLS/", "MCP/", "UI/"],
        implementation_steps=["step1", "step2", "step3"],
        warnings=[],
        raw_response="plan response",
    )


def test_scaffold_generates_child_code_and_copies_selected_components(tmp_path: Path) -> None:
    _build_repo(tmp_path)
    provider_response = json.dumps(
        {
            "files": [
                {
                    "path": "main.py",
                    "content": (
                        "#!/usr/bin/env python3\n"
                        "from agent.runtime import run\n\n"
                        "if __name__ == '__main__':\n"
                        "    print(run())\n"
                    ),
                },
                {"path": "agent/__init__.py", "content": ""},
                {"path": "agent/runtime.py", "content": "def run() -> str:\n    return 'ok'\n"},
            ]
        },
        ensure_ascii=False,
    )
    provider = FakeProvider(provider_response)
    creator = AgentCreator(provider=provider, model="gpt-4o-mini", repo_root=tmp_path)
    catalog = ComponentCatalogBuilder(tmp_path).build()

    result = creator.create_agent_project_from_plan(
        user_request="build child",
        plan=_plan("agent_child_llm"),
        catalog=catalog,
        overwrite=False,
    )

    destination = result.destination
    main_content = (destination / "main.py").read_text(encoding="utf-8")
    assert "AIDZero.py" not in main_content
    assert (destination / "agent" / "runtime.py").is_file()
    assert (destination / "TOOLS" / "AID-skill-tool" / "tool.py").is_file()
    assert not (destination / "TOOLS" / "AID-extra").exists()
    assert (destination / "UI" / "terminal" / "entrypoint.py").is_file()
    assert not (destination / "UI" / "web").exists()
    assert provider.calls

    manifest = json.loads((destination / "agent" / "child_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated_code"]["mode"] == "llm"


def test_scaffold_falls_back_to_template_when_llm_output_is_invalid(tmp_path: Path) -> None:
    _build_repo(tmp_path)
    provider = FakeProvider("this is not json")
    creator = AgentCreator(provider=provider, model="gpt-4o-mini", repo_root=tmp_path)
    catalog = ComponentCatalogBuilder(tmp_path).build()

    result = creator.create_agent_project_from_plan(
        user_request="build child",
        plan=_plan("agent_child_fallback"),
        catalog=catalog,
        overwrite=False,
    )

    destination = result.destination
    main_content = (destination / "main.py").read_text(encoding="utf-8")
    assert "Generated child agent entrypoint." in main_content
    assert "AIDZero.py" not in main_content
    assert (destination / "agent" / "runtime.py").is_file()
    assert (destination / "agent" / "provider_registry.py").is_file()

    manifest = json.loads((destination / "agent" / "child_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated_code"]["mode"] == "template_fallback"
