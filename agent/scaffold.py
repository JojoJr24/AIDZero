"""Filesystem scaffolding for parent->child agent forking."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from .generated_agent_config import AGENT_CONFIG_FILENAME, build_default_runtime_config
from .models import AgentPlan, ComponentCatalog, ScaffoldResult

FORK_ENVIRONMENT_FOLDERS = [
    "LLMProviders",
    "MCP",
    "SKILLS",
    "TOOLS",
    "UI",
    ".aidzero",
]
RUNTIME_AGENT_SUPPORT_FILES = [
    "provider_base.py",
    "openai_compatible_provider.py",
    "generated_agent_runtime.py",
]
CHILD_MANIFEST_FILENAME = "child_manifest.json"


class AgentScaffolder:
    """Creates a child agent workspace from the parent environment folders."""

    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root.resolve()

    def scaffold(
        self,
        *,
        destination: Path,
        plan: AgentPlan,
        catalog: ComponentCatalog,
        user_request: str,
        main_py_source: str | None = None,
        overwrite: bool = False,
        process_logger: Callable[[str], None] | None = None,
        process_logger_ready: Callable[[], None] | None = None,
    ) -> ScaffoldResult:
        del catalog  # The forking logic copies a fixed environment subset.

        resolved_destination = destination.resolve()
        if resolved_destination == self.source_root:
            raise ValueError("Destination cannot be the current repository root.")
        if not _is_relative_to(resolved_destination, self.source_root):
            raise ValueError(
                f"Destination must stay within repository root: {resolved_destination}"
            )

        if resolved_destination.exists() and not overwrite and any(resolved_destination.iterdir()):
            raise FileExistsError(
                f"Destination '{resolved_destination}' is not empty. Use overwrite=True to merge."
            )
        resolved_destination.mkdir(parents=True, exist_ok=True)
        if process_logger_ready is not None:
            process_logger_ready()
        _emit_log(process_logger, f"prepared destination folder: {resolved_destination}")

        result = ScaffoldResult(destination=resolved_destination)
        result.created_directories.append(resolved_destination)

        copied = self._copy_environment_folders(destination=resolved_destination, process_logger=process_logger)
        copied.extend(self._copy_runtime_agent_support(destination=resolved_destination, process_logger=process_logger))
        copied.append(
            self._write_child_manifest(
                destination=resolved_destination,
                user_request=user_request,
                plan=plan,
                process_logger=process_logger,
            )
        )
        result.copied_items = copied

        if main_py_source is not None:
            entrypoint_file = _target_in_workspace(resolved_destination, "main.py")
            entrypoint_file.write_text(main_py_source, encoding="utf-8")
            result.entrypoint_file = entrypoint_file
            _emit_log(process_logger, f"wrote generated entrypoint: {entrypoint_file}")

        runtime_config_file = _target_in_workspace(resolved_destination, AGENT_CONFIG_FILENAME)
        runtime_config_file.write_text(
            json.dumps(build_default_runtime_config(plan), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result.runtime_config_file = runtime_config_file
        _emit_log(process_logger, f"wrote runtime config: {runtime_config_file}")

        metadata_file = _target_in_workspace(resolved_destination, "agent_plan.json")
        metadata_file.write_text(
            json.dumps(plan.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result.metadata_file = metadata_file
        _emit_log(process_logger, f"wrote plan metadata: {metadata_file}")
        return result

    def _copy_environment_folders(
        self, *, destination: Path, process_logger: Callable[[str], None] | None = None
    ) -> list[Path]:
        copied: list[Path] = []
        for folder_name in FORK_ENVIRONMENT_FOLDERS:
            source_folder = self.source_root / folder_name
            if not source_folder.exists():
                _emit_log(process_logger, f"skipped missing environment folder: {source_folder}")
                continue
            target = _target_in_workspace(destination, folder_name)
            _copy_item(source_folder, target)
            copied.append(target)
            _emit_log(process_logger, f"copied environment folder: {source_folder} -> {target}")
        return copied

    def _copy_runtime_agent_support(
        self, *, destination: Path, process_logger: Callable[[str], None] | None = None
    ) -> list[Path]:
        runtime_package_dir = _target_in_workspace(destination, "agent")
        runtime_package_dir.mkdir(parents=True, exist_ok=True)
        _emit_log(process_logger, f"ensured runtime package folder: {runtime_package_dir}")

        copied: list[Path] = [runtime_package_dir]
        init_file = _target_in_workspace(destination, "agent/__init__.py")
        init_file.write_text('"""Runtime support package for generated agents."""\n', encoding="utf-8")
        copied.append(init_file)
        _emit_log(process_logger, f"wrote runtime package init file: {init_file}")

        for filename in RUNTIME_AGENT_SUPPORT_FILES:
            source_path = self.source_root / "agent" / filename
            target_path = _target_in_workspace(destination, f"agent/{filename}")
            _copy_item(source_path, target_path)
            copied.append(target_path)
            _emit_log(process_logger, f"copied runtime support file: {source_path} -> {target_path}")
        return copied

    def _write_child_manifest(
        self,
        *,
        destination: Path,
        user_request: str,
        plan: AgentPlan,
        process_logger: Callable[[str], None] | None = None,
    ) -> Path:
        context_file = _target_in_workspace(destination, f"agent/{CHILD_MANIFEST_FILENAME}")
        payload = {
            "agent_name": plan.agent_name,
            "goal": plan.goal,
            "summary": plan.summary,
            "original_user_request": user_request.strip(),
            "default_task": user_request.strip(),
            "required_llm_provider": (
                plan.required_llm_providers[0] if plan.required_llm_providers else ""
            ),
            "required_ui": list(plan.required_ui),
            "required_skills": list(plan.required_skills),
            "required_tools": list(plan.required_tools),
            "required_mcp": list(plan.required_mcp),
            "implementation_steps": list(plan.implementation_steps),
        }
        context_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _emit_log(process_logger, f"wrote child manifest: {context_file}")
        return context_file


def _target_in_workspace(workspace_root: Path, relative_path: str) -> Path:
    candidate = (workspace_root / relative_path).resolve()
    if not _is_relative_to(candidate, workspace_root):
        raise ValueError(f"Refusing to write outside workspace: {relative_path}")
    return candidate


def _copy_item(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Cannot copy missing path: {source}")
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _emit_log(process_logger: Callable[[str], None] | None, message: str) -> None:
    if process_logger is None:
        return
    process_logger(message)
