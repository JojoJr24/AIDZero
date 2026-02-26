"""Shared data models for the AIDZero runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComponentItem:
    """One discovered runtime component."""

    name: str
    path: Path
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "description": self.description,
        }


@dataclass(frozen=True)
class ComponentCatalog:
    """Repository component inventory used by planning/runtime layers."""

    root: Path
    llm_providers: list[ComponentItem]
    skills: list[ComponentItem]
    tools: list[ComponentItem]
    mcp: list[ComponentItem]
    ui: list[ComponentItem]


@dataclass(frozen=True)
class AgentPlan:
    """Normalized planning payload returned by the runtime."""

    agent_name: str
    project_folder: str
    goal: str
    summary: str
    required_llm_providers: list[str] = field(default_factory=list)
    required_skills: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    required_mcp: list[str] = field(default_factory=list)
    required_ui: list[str] = field(default_factory=list)
    folder_blueprint: list[str] = field(default_factory=list)
    implementation_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "project_folder": self.project_folder,
            "goal": self.goal,
            "summary": self.summary,
            "required_llm_providers": list(self.required_llm_providers),
            "required_skills": list(self.required_skills),
            "required_tools": list(self.required_tools),
            "required_mcp": list(self.required_mcp),
            "required_ui": list(self.required_ui),
            "folder_blueprint": list(self.folder_blueprint),
            "implementation_steps": list(self.implementation_steps),
            "warnings": list(self.warnings),
            "raw_response": self.raw_response,
        }


@dataclass(frozen=True)
class PlanningResult:
    """Planning response container used by UIs."""

    plan: AgentPlan
    catalog: ComponentCatalog
    response_text: str


@dataclass(frozen=True)
class ScaffoldResult:
    """Scaffolding output summary used by UIs."""

    destination: Path
    created_directories: list[Path]
    copied_items: list[Path]
    entrypoint_file: Path | None = None
    runtime_config_file: Path | None = None
    metadata_file: Path | None = None
    process_log_file: Path | None = None
