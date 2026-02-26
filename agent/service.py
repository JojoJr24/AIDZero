"""High-level runtime service used by terminal and web UIs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
from typing import Any

from LLMProviders.provider_base import LLMProvider
from agent.catalog import ComponentCatalogBuilder
from agent.models import AgentPlan, ComponentCatalog, ComponentItem, PlanningResult, ScaffoldResult
from agent.provider_base import InvocationTracer, ProviderBaseRuntime

_LLM_SHARED_PROVIDER_FILES = (
    Path("LLMProviders/base.py"),
    Path("LLMProviders/provider_base.py"),
    Path("LLMProviders/openai_compatible_provider.py"),
    Path("LLMProviders/__init__.py"),
    Path("LLMProviders/AGENTS.md"),
)

_MCP_SHARED_FILES = (
    Path("MCP/mcporter.json"),
    Path("MCP/run-tool-gateway.sh"),
)

_UI_SHARED_FILES = (Path("UI/AGENTS.md"),)


@dataclass(frozen=True)
class _SelectedComponents:
    providers: list[ComponentItem]
    skills: list[ComponentItem]
    tools: list[ComponentItem]
    mcp: list[ComponentItem]
    ui: list[ComponentItem]


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
        _prepare_destination(destination=destination, overwrite=overwrite)

        created_directories: list[Path] = [destination]
        copied_items: list[Path] = []

        selected_components = _select_components_for_plan(plan=plan, catalog=catalog)
        selected_paths = _paths_for_selected_components(
            repo_root=self.repo_root,
            selected=selected_components,
        )
        copied_directories, copied_targets = _copy_relative_paths(
            repo_root=self.repo_root,
            destination=destination,
            relative_paths=selected_paths,
        )
        _extend_unique(created_directories, copied_directories)
        _extend_unique(copied_items, copied_targets)

        selected_provider = _selected_provider_name(selected_components=selected_components, catalog=catalog)
        generated_files, generation_mode = self._generate_child_runtime_files(
            user_request=user_request,
            plan=plan,
            selected_components=selected_components,
            selected_provider=selected_provider,
        )
        written_directories, written_files = _write_generated_files(
            destination=destination,
            generated_files=generated_files,
        )
        _extend_unique(created_directories, written_directories)

        runtime_config_file = destination / "agent_config.json"
        runtime_config_file.write_text(
            json.dumps(
                {
                    "provider": selected_provider,
                    "model": self.model,
                    "goal": plan.goal,
                    "entrypoint": "main.py",
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
                    "selected_components": {
                        "providers": [item.name for item in selected_components.providers],
                        "skills": [item.name for item in selected_components.skills],
                        "tools": [item.name for item in selected_components.tools],
                        "mcp": [item.name for item in selected_components.mcp],
                        "ui": [item.name for item in selected_components.ui],
                    },
                    "generated_code": {
                        "mode": generation_mode,
                        "files": [str(path.relative_to(destination)) for path in written_files],
                    },
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
        process_log_file.write_text(
            _build_process_log(
                destination=destination,
                plan=plan,
                copied_items=copied_items,
                written_files=written_files,
                generation_mode=generation_mode,
            ),
            encoding="utf-8",
        )

        entrypoint_file = destination / "main.py"
        return ScaffoldResult(
            destination=destination,
            created_directories=created_directories,
            copied_items=copied_items,
            entrypoint_file=entrypoint_file if entrypoint_file.is_file() else None,
            runtime_config_file=runtime_config_file,
            metadata_file=metadata_file,
            process_log_file=process_log_file,
        )

    def _generate_child_runtime_files(
        self,
        *,
        user_request: str,
        plan: AgentPlan,
        selected_components: _SelectedComponents,
        selected_provider: str | None,
    ) -> tuple[dict[str, str], str]:
        system_prompt = (
            "You generate source files for a child Python agent project.\n"
            "Return ONLY one JSON object with this exact schema:\n"
            '{"files":[{"path":"relative/path.py","content":"full file content"}]}\n'
            "Rules:\n"
            "- Include a root `main.py` file as the entrypoint.\n"
            "- Put runtime logic under `agent/` (for example `agent/runtime.py`).\n"
            "- Do not reference parent files (`AIDZero.py`, parent paths, or parent imports).\n"
            "- Reuse inherited scaffolding folders conceptually, but generate fresh child code.\n"
            "- Output raw JSON only (no markdown, no prose)."
        )
        request_payload = {
            "user_request": user_request,
            "plan": plan.to_dict(),
            "selected_components": {
                "providers": [item.name for item in selected_components.providers],
                "skills": [item.name for item in selected_components.skills],
                "tools": [item.name for item in selected_components.tools],
                "mcp": [item.name for item in selected_components.mcp],
                "ui": [item.name for item in selected_components.ui],
            },
            "runtime_defaults": {
                "provider": selected_provider,
                "model": self.model,
            },
            "required_files": [
                "main.py",
                "agent/__init__.py",
                "agent/runtime.py",
            ],
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Generate child files now using this context:\n"
                + json.dumps(request_payload, indent=2, ensure_ascii=False),
            },
        ]

        llm_output = ""
        try:
            payload = self.provider.chat(self.model, messages)
            llm_output = _extract_response_text(payload).strip()
        except Exception:  # noqa: BLE001
            llm_output = ""

        parsed_files = _parse_generated_files_payload(llm_output)
        if parsed_files is not None:
            return parsed_files, "llm"

        return (
            _default_generated_child_files(
                goal=plan.goal,
                provider_name=selected_provider or "AID-google_gemini",
                model_name=self.model or "gemini-2.5-flash",
            ),
            "template_fallback",
        )


def _prepare_destination(*, destination: Path, overwrite: bool) -> None:
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


def _select_components_for_plan(*, plan: AgentPlan, catalog: ComponentCatalog) -> _SelectedComponents:
    providers = _resolve_required_components(plan.required_llm_providers, catalog.llm_providers)
    if not providers and catalog.llm_providers:
        providers = [catalog.llm_providers[0]]
    return _SelectedComponents(
        providers=providers,
        skills=_resolve_required_components(plan.required_skills, catalog.skills),
        tools=_resolve_required_components(plan.required_tools, catalog.tools),
        mcp=_resolve_required_components(plan.required_mcp, catalog.mcp),
        ui=_resolve_required_components(plan.required_ui, catalog.ui),
    )


def _resolve_required_components(
    required_names: list[str],
    available: list[ComponentItem],
) -> list[ComponentItem]:
    if not required_names or not available:
        return []

    exact_map = {item.name: item for item in available}
    alias_map = {_component_alias(item.name): item for item in available}
    selected: list[ComponentItem] = []
    seen: set[str] = set()
    for raw_name in required_names:
        name = raw_name.strip()
        if not name:
            continue
        item = exact_map.get(name)
        if item is None:
            item = alias_map.get(_component_alias(name))
        if item is None or item.name in seen:
            continue
        selected.append(item)
        seen.add(item.name)
    return selected


def _component_alias(name: str) -> str:
    lowered = name.strip().lower()
    if lowered.startswith("aid-"):
        return lowered[4:]
    return lowered


def _selected_provider_name(*, selected_components: _SelectedComponents, catalog: ComponentCatalog) -> str | None:
    if selected_components.providers:
        return selected_components.providers[0].name
    if catalog.llm_providers:
        return catalog.llm_providers[0].name
    return None


def _paths_for_selected_components(
    *,
    repo_root: Path,
    selected: _SelectedComponents,
) -> list[Path]:
    paths: set[Path] = set()
    for item in selected.providers:
        paths.add(item.path)
    for item in selected.skills:
        paths.add(item.path)
    for item in selected.tools:
        paths.add(item.path)
    for item in selected.mcp:
        paths.add(item.path)
    for item in selected.ui:
        paths.add(item.path)

    if selected.providers:
        for shared in _LLM_SHARED_PROVIDER_FILES:
            if (repo_root / shared).exists():
                paths.add(shared)
    if selected.mcp:
        for shared in _MCP_SHARED_FILES:
            if (repo_root / shared).exists():
                paths.add(shared)
    if selected.ui:
        for shared in _UI_SHARED_FILES:
            if (repo_root / shared).exists():
                paths.add(shared)

    return sorted(paths, key=lambda path: str(path))


def _copy_relative_paths(
    *,
    repo_root: Path,
    destination: Path,
    relative_paths: list[Path],
) -> tuple[list[Path], list[Path]]:
    created_directories: list[Path] = []
    copied_targets: list[Path] = []
    for relative_path in relative_paths:
        source = repo_root / relative_path
        if not source.exists():
            continue
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
            _append_unique(created_directories, target)
        else:
            shutil.copy2(source, target)
        _append_unique(copied_targets, target)
    return created_directories, copied_targets


def _write_generated_files(
    *,
    destination: Path,
    generated_files: dict[str, str],
) -> tuple[list[Path], list[Path]]:
    created_directories: list[Path] = []
    written_files: list[Path] = []
    for relative_path, content in generated_files.items():
        safe_relative = _safe_output_relative_path(relative_path)
        if safe_relative is None:
            continue
        target = destination / safe_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.parent != destination:
            _append_unique(created_directories, target.parent)
        normalized_content = content if content.endswith("\n") else content + "\n"
        target.write_text(normalized_content, encoding="utf-8")
        _append_unique(written_files, target)

    if not written_files:
        raise RuntimeError("No child code files were generated for the destination project.")
    return created_directories, written_files


def _safe_output_relative_path(raw_path: str) -> Path | None:
    normalized = raw_path.strip().replace("\\", "/")
    if not normalized:
        return None
    candidate = Path(normalized)
    if candidate.is_absolute():
        return None
    if ".." in candidate.parts:
        return None
    parts = [part for part in candidate.parts if part and part != "."]
    if not parts:
        return None
    return Path(*parts)


def _build_process_log(
    *,
    destination: Path,
    plan: AgentPlan,
    copied_items: list[Path],
    written_files: list[Path],
    generation_mode: str,
) -> str:
    timestamp = datetime.now(UTC).isoformat()
    lines = [
        f"{timestamp} | step1 plan defined: {plan.agent_name}",
        f"{timestamp} | step2 scaffold copied: {len(copied_items)} paths",
        (
            f"{timestamp} | step3 child code generated: {generation_mode}, "
            f"{len(written_files)} files"
        ),
        f"{timestamp} | scaffold created at {destination}",
    ]
    return "\n".join(lines) + "\n"


def _parse_generated_files_payload(raw_text: str) -> dict[str, str] | None:
    if not raw_text.strip():
        return None
    payload = _extract_json_object(raw_text)
    if payload is None:
        return None
    files_payload = payload.get("files")
    if not isinstance(files_payload, list):
        return None

    files: dict[str, str] = {}
    for item in files_payload:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        content = item.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        safe_relative = _safe_output_relative_path(path)
        if safe_relative is None:
            continue
        files[safe_relative.as_posix()] = content

    if not _has_minimum_child_files(files):
        return None
    return files


def _has_minimum_child_files(files: dict[str, str]) -> bool:
    if "main.py" not in files:
        return False
    return any(path.startswith("agent/") and path.endswith(".py") for path in files)


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    candidates: list[str] = []
    stripped = raw_text.strip()
    if stripped:
        candidates.append(stripped)

    code_fence_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
    for match in code_fence_pattern.finditer(raw_text):
        candidates.insert(0, match.group(1).strip())

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw_text[start : end + 1].strip())

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                chunks: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            chunks.append(text)
                if chunks:
                    return "\n".join(chunks)

    content_blocks = payload.get("content")
    if isinstance(content_blocks, list):
        chunks = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)

    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            chunks = [
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            if chunks:
                return "\n".join(chunks)
    return ""


def _default_generated_child_files(
    *,
    goal: str,
    provider_name: str,
    model_name: str,
) -> dict[str, str]:
    goal_literal = json.dumps(goal, ensure_ascii=False)
    provider_literal = json.dumps(provider_name, ensure_ascii=False)
    model_literal = json.dumps(model_name, ensure_ascii=False)

    main_py = f"""#!/usr/bin/env python3
\"\"\"Generated child agent entrypoint.\"\"\"

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.runtime import ChildAgentRuntime, load_runtime_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=\"Generated child agent runtime.\")
    parser.add_argument(\"--request\", default=None, help=\"Task request for the child agent.\")
    parser.add_argument(\"--provider\", default=None, help=\"Provider override.\")
    parser.add_argument(\"--model\", default=None, help=\"Model override.\")
    return parser.parse_args()


def _resolve(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    normalized = value.strip()
    return normalized or fallback


def main() -> int:
    args = _parse_args()
    config = load_runtime_config(ROOT / \"agent_config.json\")
    provider = _resolve(args.provider, str(config.get(\"provider\") or {provider_literal}))
    model = _resolve(args.model, str(config.get(\"model\") or {model_literal}))
    default_request = str(config.get(\"goal\") or {goal_literal}).strip()
    request = _resolve(args.request, default_request).strip()
    if not request:
        print(\"error> empty request. Use --request or set goal in agent_config.json.\")
        return 2

    runtime = ChildAgentRuntime(
        repo_root=ROOT,
        provider_name=provider,
        model=model,
        goal=str(config.get(\"goal\") or {goal_literal}),
    )
    print(runtime.run(request))
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())
"""

    runtime_py = """from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from agent.provider_registry import ProviderRegistry


def load_runtime_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding=\"utf-8\"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


@dataclass
class ChildAgentRuntime:
    repo_root: Path
    provider_name: str
    model: str
    goal: str

    def run(self, user_request: str) -> str:
        provider = ProviderRegistry(self.repo_root).create(self.provider_name)
        prompt = self._build_prompt(user_request)
        answer = provider.generate_text(self.model, prompt)
        if isinstance(answer, str):
            return answer.strip()
        return str(answer)

    def _build_prompt(self, user_request: str) -> str:
        goal_text = self.goal.strip()
        request_text = user_request.strip()
        if goal_text:
            return (
                \"You are a generated child agent.\\\\n\"
                \"Primary goal:\\\\n\"
                f\"{goal_text}\\\\n\\\\n\"
                \"User request:\\\\n\"
                f\"{request_text}\\\\n\\\\n\"
                \"Provide a direct and useful response.\"
            )
        return (
            \"You are a generated child agent.\\\\n\"
            \"User request:\\\\n\"
            f\"{request_text}\\\\n\\\\n\"
            \"Provide a direct and useful response.\"
        )
"""

    provider_registry_py = """from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

from LLMProviders.provider_base import LLMProvider


class ProviderRegistry:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.providers_root = self.repo_root / \"LLMProviders\"

    def names(self) -> list[str]:
        names: list[str] = []
        if not self.providers_root.is_dir():
            return names
        for entry in sorted(self.providers_root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir() or not entry.name.startswith(\"AID-\"):
                continue
            if (entry / \"provider.py\").is_file():
                names.append(entry.name)
        return names

    def create(self, provider_name: str) -> LLMProvider:
        normalized_name = provider_name.strip()
        if not normalized_name:
            raise ValueError(\"provider_name cannot be empty.\")
        module_path = self.providers_root / normalized_name / \"provider.py\"
        if not module_path.is_file():
            raise FileNotFoundError(f\"Provider module not found: {module_path}\")

        module_name = f\"child_provider_{normalized_name.replace('-', '_')}\"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f\"Unable to load provider module: {module_path}\")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        provider_cls = _find_provider_class(module)
        return provider_cls()


def _find_provider_class(module: Any) -> type[LLMProvider]:
    candidates: list[type[Any]] = []
    for _, member in inspect.getmembers(module, inspect.isclass):
        if member.__module__ != module.__name__:
            continue
        if not member.__name__.endswith(\"Provider\"):
            continue
        if member.__name__ == \"OpenAICompatibleProvider\":
            continue
        candidates.append(member)
    if not candidates:
        raise RuntimeError(f\"No provider class found in module '{module.__name__}'.\")
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda item: item.__name__)
    return candidates[0]
"""

    return {
        "main.py": main_py,
        "agent/__init__.py": "\"\"\"Generated child agent package.\"\"\"\n",
        "agent/runtime.py": runtime_py,
        "agent/provider_registry.py": provider_registry_py,
    }


def _append_unique(target: list[Path], value: Path) -> None:
    for existing in target:
        if existing == value:
            return
    target.append(value)


def _extend_unique(target: list[Path], values: list[Path]) -> None:
    for value in values:
        _append_unique(target, value)


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
            "Step 1: define the agent plan.",
            "Step 2: copy selected parent scaffolding into the child workspace.",
            "Step 3: generate fresh child code with root main.py and agent/* runtime logic.",
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
