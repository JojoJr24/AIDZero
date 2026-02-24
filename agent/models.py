"""Data models for cataloging, planning, and scaffolding agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ComponentItem:
    """Single reusable repository component."""

    name: str
    path: Path
    kind: str
    description: str | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "kind": self.kind,
            "path": self.path.as_posix(),
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass
class ComponentCatalog:
    """Inventory of reusable modules in the current repository."""

    root: Path
    llm_providers: list[ComponentItem] = field(default_factory=list)
    skills: list[ComponentItem] = field(default_factory=list)
    tools: list[ComponentItem] = field(default_factory=list)
    mcp: list[ComponentItem] = field(default_factory=list)
    ui: list[ComponentItem] = field(default_factory=list)

    def as_prompt_payload(self) -> dict[str, Any]:
        return {
            "llm_providers": [item.to_prompt_dict() for item in self.llm_providers],
            "skills": [item.to_prompt_dict() for item in self.skills],
            "tools": [item.to_prompt_dict() for item in self.tools],
            "mcp": [item.to_prompt_dict() for item in self.mcp],
            "ui": [item.to_prompt_dict() for item in self.ui],
        }

    def names_by_kind(self) -> dict[str, set[str]]:
        return {
            "llm_providers": {item.name for item in self.llm_providers},
            "skills": {item.name for item in self.skills},
            "tools": {item.name for item in self.tools},
            "mcp": {item.name for item in self.mcp},
            "ui": {item.name for item in self.ui},
        }


@dataclass
class AgentPlan:
    """Structured plan produced by the planning model."""

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
        payload = asdict(self)
        payload["raw_response"] = self.raw_response
        return payload


@dataclass
class ScaffoldResult:
    """Output from project scaffolding/copying."""

    destination: Path
    created_directories: list[Path] = field(default_factory=list)
    copied_items: list[Path] = field(default_factory=list)
    entrypoint_file: Path | None = None
    runtime_config_file: Path | None = None
    metadata_file: Path | None = None
