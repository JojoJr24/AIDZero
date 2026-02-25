"""High-level orchestration service for creating new agents."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Any

from .catalog import ComponentCatalogBuilder
from .entrypoint_writer import AgentEntrypointWriter
from .models import AgentPlan, ComponentCatalog, ScaffoldResult
from .planner import AgentPlanner
from .runtime_config import RuntimeConfigStore
from .scaffold import AgentScaffolder

PROCESS_LOG_FILENAME = "generation_process.log"


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
        generation_process_log_enabled: bool | None = None,
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
        self._generation_process_log_enabled = generation_process_log_enabled

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

        This forks a child workspace by copying only the parent environment folders
        (LLMProviders, MCP, SKILLS, TOOLS, UI, .aidzero), then writes generated
        child files (main.py and agent/* runtime/context files).
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
        """Generate main.py and scaffold the destination child-agent fork."""
        destination = self.resolve_destination_from_plan(plan)
        process_logger = _GenerationProcessLogger(
            destination / PROCESS_LOG_FILENAME,
            enabled=self._resolve_process_log_enabled(),
        )
        process_logger.log(f"generation started for agent '{plan.agent_name}'")
        process_logger.log(f"user request: {user_request.strip()}")
        process_logger.log(f"project folder: {plan.project_folder}")
        process_logger.log(
            "selected components: "
            f"providers={plan.required_llm_providers}, "
            f"skills={plan.required_skills}, "
            f"tools={plan.required_tools}, "
            f"mcp={plan.required_mcp}, "
            f"ui={plan.required_ui}"
        )
        if plan.warnings:
            process_logger.log(f"plan warnings: {json.dumps(plan.warnings, ensure_ascii=False)}")

        try:
            process_logger.log("generating child entrypoint source (main.py)")
            main_py_source = self.entrypoint_writer.generate_main_py(
                user_request=user_request,
                plan=plan,
            )
            process_logger.log("entrypoint generation completed")

            process_logger.log("starting filesystem scaffold")
            scaffold_result = self.scaffolder.scaffold(
                destination=destination,
                plan=plan,
                catalog=catalog,
                user_request=user_request,
                main_py_source=main_py_source,
                overwrite=overwrite,
                process_logger=process_logger.log,
                process_logger_ready=process_logger.activate,
            )
            scaffold_result.process_log_file = process_logger.log_file
            process_logger.log(
                "scaffold completed successfully: "
                f"created_directories={len(scaffold_result.created_directories)}, "
                f"copied_items={len(scaffold_result.copied_items)}"
            )
            return scaffold_result
        except Exception as error:  # noqa: BLE001
            process_logger.log(f"generation failed: {error}")
            process_logger.log(traceback.format_exc().rstrip())
            process_logger.activate()
            raise

    def resolve_destination_from_plan(self, plan: AgentPlan) -> Path:
        """Resolve a safe destination path from plan.project_folder."""
        sanitized = _sanitize_relative_path(plan.project_folder)
        return self.repo_root / sanitized

    def _resolve_process_log_enabled(self) -> bool:
        if self._generation_process_log_enabled is not None:
            return self._generation_process_log_enabled
        try:
            config = RuntimeConfigStore(self.repo_root).load()
        except Exception:  # noqa: BLE001
            return True
        if config is None:
            return True
        return bool(config.generation_process_log_enabled)


def _sanitize_relative_path(path_value: str) -> Path:
    candidate = path_value.strip().strip("/")
    if not candidate:
        raise ValueError("Plan project_folder is empty.")
    parts = [part for part in candidate.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe project_folder in plan: {path_value}")
    return Path(*parts)


class _GenerationProcessLogger:
    """Appends timestamped generation events to a file inside the child project."""

    def __init__(self, log_file: Path, *, enabled: bool = True) -> None:
        self.log_file = log_file if enabled else None
        self._enabled = enabled
        self._active = False
        self._buffer: list[tuple[str, str]] = []

    def activate(self) -> None:
        if not self._enabled:
            return
        if self._active:
            return
        assert self.log_file is not None
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as handle:
            for timestamp, message in self._buffer:
                handle.write(f"{timestamp} | {message}\n")
        self._buffer = []
        self._active = True

    def log(self, message: str) -> None:
        if not self._enabled:
            return
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self._active:
            self._buffer.append((timestamp, message))
            return
        assert self.log_file is not None
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} | {message}\n")
