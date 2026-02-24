"""Filesystem scaffolding and selective component copying."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .models import AgentPlan, ComponentCatalog, ScaffoldResult


DEFAULT_BLUEPRINT = ["LLMProviders", "MCP", "SKILLS", "TOOLS", "UI", "src", "tests"]


class AgentScaffolder:
    """Creates a new agent project and copies selected reusable components."""

    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root.resolve()

    def scaffold(
        self,
        *,
        destination: Path,
        plan: AgentPlan,
        catalog: ComponentCatalog,
        main_py_source: str | None = None,
        overwrite: bool = False,
    ) -> ScaffoldResult:
        resolved_destination = destination.resolve()
        if resolved_destination.exists() and not overwrite and any(resolved_destination.iterdir()):
            raise FileExistsError(
                f"Destination '{resolved_destination}' is not empty. Use overwrite=True to merge."
            )
        resolved_destination.mkdir(parents=True, exist_ok=True)

        result = ScaffoldResult(destination=resolved_destination)
        blueprint = plan.folder_blueprint or list(DEFAULT_BLUEPRINT)
        for folder in blueprint:
            target_dir = resolved_destination / folder
            target_dir.mkdir(parents=True, exist_ok=True)
            result.created_directories.append(target_dir)

        copied: list[Path] = []
        copied.extend(self._copy_selected_group("LLMProviders", plan.required_llm_providers, catalog, resolved_destination))
        copied.extend(self._copy_selected_group("SKILLS", plan.required_skills, catalog, resolved_destination))
        copied.extend(self._copy_selected_group("TOOLS", plan.required_tools, catalog, resolved_destination))
        copied.extend(self._copy_selected_group("MCP", plan.required_mcp, catalog, resolved_destination))
        copied.extend(self._copy_selected_group("UI", plan.required_ui, catalog, resolved_destination))
        result.copied_items = copied

        if main_py_source is not None:
            entrypoint_file = resolved_destination / "main.py"
            entrypoint_file.write_text(main_py_source, encoding="utf-8")
            result.entrypoint_file = entrypoint_file

        metadata_file = resolved_destination / "agent_plan.json"
        with metadata_file.open("w", encoding="utf-8") as handle:
            json.dump(plan.to_dict(), handle, indent=2, ensure_ascii=False)
        result.metadata_file = metadata_file
        return result

    def _copy_selected_group(
        self,
        folder_name: str,
        selected_items: list[str],
        catalog: ComponentCatalog,
        destination: Path,
    ) -> list[Path]:
        if not selected_items:
            return []

        source_paths = self._build_source_path_map(catalog, folder_name)
        copied: list[Path] = []
        for item_name in selected_items:
            source_rel = source_paths.get(item_name)
            if source_rel is None:
                continue
            source_path = self.source_root / source_rel
            target_path = destination / source_rel
            _copy_item(source_path, target_path)
            copied.append(target_path)
        return copied

    @staticmethod
    def _build_source_path_map(catalog: ComponentCatalog, folder_name: str) -> dict[str, Path]:
        if folder_name == "LLMProviders":
            return {item.name: item.path for item in catalog.llm_providers}
        if folder_name == "SKILLS":
            return {item.name: item.path for item in catalog.skills}
        if folder_name == "TOOLS":
            return {item.name: item.path for item in catalog.tools}
        if folder_name == "MCP":
            return {item.name: item.path for item in catalog.mcp}
        if folder_name == "UI":
            return {item.name: item.path for item in catalog.ui}
        return {}


def _copy_item(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Cannot copy missing path: {source}")
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
