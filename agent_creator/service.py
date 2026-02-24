"""High-level orchestration service for creating new agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import ComponentCatalogBuilder
from .entrypoint_writer import AgentEntrypointWriter
from .models import AgentPlan, ComponentCatalog, ScaffoldResult
from .planner import AgentPlanner
from .scaffold import AgentScaffolder


@dataclass
class AgentCreationResult:
    """Combined output of planning and scaffolding."""

    catalog: ComponentCatalog
    plan: AgentPlan
    scaffold: ScaffoldResult | None = None


class AgentCreator:
    """Core agent that plans and builds a new agent project."""

    def __init__(
        self,
        *,
        provider: Any,
        model: str,
        repo_root: Path | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.provider = provider
        self.model = model
        self.catalog_builder = ComponentCatalogBuilder(self.repo_root)
        self.planner = AgentPlanner(provider=provider, model=model, temperature=temperature)
        self.entrypoint_writer = AgentEntrypointWriter(
            provider=provider,
            model=model,
            temperature=temperature,
        )
        self.scaffolder = AgentScaffolder(self.repo_root)

    def scan_components(self) -> ComponentCatalog:
        """Read all reusable components from repository folders."""
        return self.catalog_builder.build()

    def describe_requirements(self, user_request: str) -> AgentCreationResult:
        """Ask the LLM to describe what the requested agent needs."""
        catalog = self.scan_components()
        plan = self.planner.plan(user_request=user_request, catalog=catalog)
        return AgentCreationResult(catalog=catalog, plan=plan)

    def create_agent_project(
        self,
        *,
        user_request: str,
        overwrite: bool = False,
    ) -> AgentCreationResult:
        """
        Plan and scaffold a new agent project.

        This creates a folder blueprint and copies selected providers/tools/skills/MCP/UI modules.
        """
        result = self.describe_requirements(user_request=user_request)
        scaffold = self.create_agent_project_from_plan(
            user_request=user_request,
            plan=result.plan,
            catalog=result.catalog,
            overwrite=overwrite,
        )
        result.scaffold = scaffold
        return result

    def create_agent_project_from_plan(
        self,
        *,
        user_request: str,
        plan: AgentPlan,
        catalog: ComponentCatalog,
        overwrite: bool = False,
    ) -> ScaffoldResult:
        """Generate main.py via LLM, then scaffold project folders and copy selected modules."""
        destination = self.resolve_destination_from_plan(plan)
        main_py_source = self.entrypoint_writer.generate_main_py(
            user_request=user_request,
            plan=plan,
        )
        return self.scaffolder.scaffold(
            destination=destination,
            plan=plan,
            catalog=catalog,
            main_py_source=main_py_source,
            overwrite=overwrite,
        )

    def resolve_destination_from_plan(self, plan: AgentPlan) -> Path:
        """Resolve a safe destination path from plan.project_folder."""
        sanitized = _sanitize_relative_path(plan.project_folder)
        return self.repo_root / sanitized


def _sanitize_relative_path(path_value: str) -> Path:
    candidate = path_value.strip().strip("/")
    if not candidate:
        raise ValueError("Plan project_folder is empty.")
    parts = [part for part in candidate.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe project_folder in plan: {path_value}")
    return Path(*parts)
