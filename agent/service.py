"""High-level runtime service used by terminal and web UIs."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from datetime import UTC, datetime

from LLMProviders.provider_base import LLMProvider
from agent.catalog import ComponentCatalogBuilder
from agent.models import AgentPlan, ComponentCatalog, PlanningResult, ScaffoldResult
from agent.provider_base import InvocationTracer, ProviderBaseRuntime


class AgentCreator:
    """Core orchestrator invoked by UIs."""

    def __init__(self, *, provider: LLMProvider, model: str, repo_root: Path) -> None:
        self.provider = provider
        self.model = model.strip()
        self.repo_root = repo_root.resolve()

    def scan_components(self) -> ComponentCatalog:
        return ComponentCatalogBuilder(self.repo_root).build()

    def respond_to_prompt(
        self,
        *,
        user_request: str,
        ui_name: str | None = None,
        invocation_tracer: InvocationTracer | None = None,
    ) -> str:
        catalog = self.scan_components()
        runtime = ProviderBaseRuntime(
            provider=self.provider,
            model=self.model,
            repo_root=self.repo_root,
            catalog=catalog,
        )
        return runtime.ask(prompt=user_request, ui_name=ui_name, invocation_tracer=invocation_tracer)

    def describe_requirements(
        self,
        *,
        user_request: str,
        ui_name: str | None = None,
        invocation_tracer: InvocationTracer | None = None,
    ) -> PlanningResult:
        catalog = self.scan_components()
        response_text = self.respond_to_prompt(
            user_request=user_request,
            ui_name=ui_name,
            invocation_tracer=invocation_tracer,
        )
        plan = _build_plan_from_response(
            user_request=user_request,
            response_text=response_text,
            catalog=catalog,
        )
        return PlanningResult(plan=plan, catalog=catalog, response_text=response_text)

    def create_agent_project_from_plan(
        self,
        *,
        user_request: str,
        plan: AgentPlan,
        catalog: ComponentCatalog,
        overwrite: bool = False,
    ) -> ScaffoldResult:
        destination = self.repo_root / "generated_agents" / plan.project_folder
        if destination.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Destination already exists: {destination}. Use overwrite=True to continue."
                )
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        destination.mkdir(parents=True, exist_ok=False)

        created_directories: list[Path] = [destination]
        copied_items: list[Path] = []

        for component_dir in ("LLMProviders", "TOOLS", "SKILLS", "MCP", "UI"):
            source = self.repo_root / component_dir
            target = destination / component_dir
            if not source.exists():
                continue
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
                created_directories.append(target)
            else:
                shutil.copy2(source, target)
            copied_items.append(target)

        main_file = destination / "main.py"
        main_file.write_text(
            (
                "#!/usr/bin/env python3\n"
                '"""Generated child runtime bootstrap."""\n\n'
                "from __future__ import annotations\n\n"
                "from pathlib import Path\n"
                "import subprocess\n"
                "import sys\n\n"
                "ROOT = Path(__file__).resolve().parent\n\n"
                "def main() -> int:\n"
                "    cmd = [sys.executable, str(ROOT / 'AIDZero.py'), *sys.argv[1:]]\n"
                "    return subprocess.run(cmd, cwd=ROOT).returncode\n\n"
                "if __name__ == '__main__':\n"
                "    raise SystemExit(main())\n"
            ),
            encoding="utf-8",
        )

        runtime_config_file = destination / "agent_config.json"
        runtime_config_file.write_text(
            json.dumps(
                {
                    "provider": plan.required_llm_providers[0] if plan.required_llm_providers else None,
                    "model": self.model,
                    "goal": plan.goal,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        metadata_file = destination / "agent" / "child_manifest.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.write_text(
            json.dumps(
                {
                    "request": user_request,
                    "plan": plan.to_dict(),
                    "catalog_counts": {
                        "providers": len(catalog.llm_providers),
                        "skills": len(catalog.skills),
                        "tools": len(catalog.tools),
                        "mcp": len(catalog.mcp),
                        "ui": len(catalog.ui),
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        process_log_file = destination / "generation_process.log"
        timestamp = datetime.now(UTC).isoformat()
        process_log_file.write_text(
            (
                f"{timestamp} | request received\n"
                f"{timestamp} | plan generated: {plan.agent_name}\n"
                f"{timestamp} | scaffold created at {destination}\n"
            ),
            encoding="utf-8",
        )
        return ScaffoldResult(
            destination=destination,
            created_directories=created_directories,
            copied_items=copied_items,
            entrypoint_file=main_file,
            runtime_config_file=runtime_config_file,
            metadata_file=metadata_file,
            process_log_file=process_log_file,
        )


def _build_plan_from_response(
    *,
    user_request: str,
    response_text: str,
    catalog: ComponentCatalog,
) -> AgentPlan:
    normalized_request = user_request.strip()
    project_folder = _slugify(normalized_request)[:48] or "agent_project"
    if not project_folder.startswith("agent_"):
        project_folder = f"agent_{project_folder}"
    agent_name = "".join(part.capitalize() for part in project_folder.split("_"))
    summary = _summarize_response_text(response_text)
    return AgentPlan(
        agent_name=agent_name,
        project_folder=project_folder,
        goal=normalized_request,
        summary=summary,
        required_llm_providers=[item.name for item in catalog.llm_providers],
        required_skills=[item.name for item in catalog.skills],
        required_tools=[item.name for item in catalog.tools],
        required_mcp=[item.name for item in catalog.mcp],
        required_ui=[item.name for item in catalog.ui],
        folder_blueprint=[
            "agent/",
            "LLMProviders/",
            "TOOLS/",
            "SKILLS/",
            "MCP/",
            "UI/",
        ],
        implementation_steps=[
            "Read user request from UI.",
            "Invoke provider base with tools + MCP + skills loaded.",
            "Return final response and persist minimal runtime metadata.",
        ],
        warnings=[],
        raw_response=response_text,
    )


def _slugify(value: str) -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered)
    return lowered.strip("_")


def _summarize_response_text(response_text: str, *, max_chars: int = 600) -> str:
    normalized = response_text.strip()
    if not normalized:
        return "No response produced."

    compact = re.sub(r"\s+", " ", normalized)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."
