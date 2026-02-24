"""Planning logic that asks an LLM which components a new agent needs."""

from __future__ import annotations

import json
from typing import Any

from .models import AgentPlan, ComponentCatalog


DEFAULT_FOLDER_BLUEPRINT = ["LLMProviders", "MCP", "SKILLS", "TOOLS", "UI", "src", "tests"]
DEFAULT_PROJECT_ROOT = "generated_agents"


class AgentPlanner:
    """Builds a structured plan from a user requirement and component catalog."""

    def __init__(self, provider: Any, model: str, *, temperature: float = 0.1) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature

    def plan(self, user_request: str, catalog: ComponentCatalog) -> AgentPlan:
        if not user_request.strip():
            raise ValueError("user_request cannot be empty.")

        prompt = self._build_prompt(user_request=user_request, catalog=catalog)
        raw_text = self.provider.generate_text(
            model=self.model,
            prompt=prompt,
            generation_config={"temperature": self.temperature},
        )

        parsed = _parse_json_payload(raw_text)
        validated = self._validate_against_catalog(parsed, catalog)
        validated.raw_response = raw_text
        return validated

    def _build_prompt(self, *, user_request: str, catalog: ComponentCatalog) -> str:
        payload = catalog.as_prompt_payload()
        return (
            "You are AIDZero's planning engine. "
            "Analyze the user request and decide which repository components are required.\n\n"
            "Return ONLY valid JSON. Do not include markdown, explanation, or extra text.\n"
            "JSON schema:\n"
            "{\n"
            '  "agent_name": "short-kebab-or-snake-name",\n'
            '  "project_folder": "relative-folder-path-where-agent-will-be-created",\n'
            '  "goal": "1-2 sentences",\n'
            '  "summary": "high-level reasoning in <=120 words",\n'
            '  "required_llm_providers": ["name_from_catalog"],\n'
            '  "required_skills": ["name_from_catalog"],\n'
            '  "required_tools": ["name_from_catalog"],\n'
            '  "required_mcp": ["name_from_catalog"],\n'
            '  "required_ui": ["name_from_catalog"],\n'
            '  "folder_blueprint": ["LLMProviders","MCP","SKILLS","TOOLS","UI","src","tests"],\n'
            '  "implementation_steps": ["short actionable steps"]\n'
            "}\n\n"
            "Rules:\n"
            "- project_folder must be a relative path and should usually start with generated_agents/.\n"
            "- Select only component names that exist in the catalog.\n\n"
            f"User request:\n{user_request.strip()}\n\n"
            f"Available catalog:\n{json.dumps(payload, indent=2)}\n"
        )

    def _validate_against_catalog(self, parsed: dict[str, Any], catalog: ComponentCatalog) -> AgentPlan:
        names = catalog.names_by_kind()
        warnings: list[str] = []

        required_llm_providers = _filter_known(
            parsed.get("required_llm_providers", []), names["llm_providers"], "required_llm_providers", warnings
        )
        required_skills = _filter_known(
            parsed.get("required_skills", []), names["skills"], "required_skills", warnings
        )
        required_tools = _filter_known(
            parsed.get("required_tools", []), names["tools"], "required_tools", warnings
        )
        required_mcp = _filter_known(
            parsed.get("required_mcp", []), names["mcp"], "required_mcp", warnings
        )
        required_ui = _filter_known(
            parsed.get("required_ui", []), names["ui"], "required_ui", warnings
        )

        folder_blueprint = _normalize_string_list(parsed.get("folder_blueprint", []))
        if not folder_blueprint:
            folder_blueprint = list(DEFAULT_FOLDER_BLUEPRINT)

        implementation_steps = _normalize_string_list(parsed.get("implementation_steps", []))

        agent_name = parsed.get("agent_name")
        if not isinstance(agent_name, str) or not agent_name.strip():
            agent_name = "generated_agent"
            warnings.append("Planner response missing agent_name; defaulted to 'generated_agent'.")
        normalized_agent_name = agent_name.strip()

        project_folder = _normalize_project_folder(parsed.get("project_folder"))
        if project_folder is None:
            project_folder = f"{DEFAULT_PROJECT_ROOT}/{_slugify(normalized_agent_name)}"
            warnings.append(
                "Planner response missing/invalid project_folder; defaulted to "
                f"'{project_folder}'."
            )

        goal = parsed.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            goal = "No explicit goal provided."
            warnings.append("Planner response missing goal; defaulted to fallback text.")

        summary = parsed.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = "No summary provided by planner."
            warnings.append("Planner response missing summary; defaulted to fallback text.")

        return AgentPlan(
            agent_name=normalized_agent_name,
            project_folder=project_folder,
            goal=goal.strip(),
            summary=summary.strip(),
            required_llm_providers=required_llm_providers,
            required_skills=required_skills,
            required_tools=required_tools,
            required_mcp=required_mcp,
            required_ui=required_ui,
            folder_blueprint=folder_blueprint,
            implementation_steps=implementation_steps,
            warnings=warnings,
        )


def _filter_known(
    values: Any,
    known_names: set[str],
    label: str,
    warnings: list[str],
) -> list[str]:
    selected = _normalize_string_list(values)
    filtered: list[str] = []
    for name in selected:
        if name in known_names:
            filtered.append(name)
            continue
        warnings.append(f"Ignored unknown {label} entry: {name}")
    return filtered


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                cleaned.append(stripped)
    return cleaned


def _normalize_project_folder(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().strip("/").strip()
    if not cleaned:
        return None
    if cleaned.startswith(".") or cleaned.startswith("~"):
        return None
    parts = [part for part in cleaned.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        return None
    return "/".join(parts)


def _slugify(value: str) -> str:
    lowered = value.lower().strip()
    result_chars: list[str] = []
    previous_dash = False
    for char in lowered:
        if char.isalnum():
            result_chars.append(char)
            previous_dash = False
            continue
        if not previous_dash:
            result_chars.append("-")
            previous_dash = True
    slug = "".join(result_chars).strip("-")
    return slug or "generated-agent"


def _parse_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    for candidate in (_try_full_json(stripped), _try_code_fence_json(stripped), _try_braced_json(stripped)):
        if candidate is not None:
            return candidate
    raise ValueError(f"Planner did not return valid JSON. Raw output: {text[:600]}")


def _try_full_json(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _try_code_fence_json(text: str) -> dict[str, Any] | None:
    marker = "```"
    if marker not in text:
        return None
    segments = text.split(marker)
    for segment in segments:
        candidate = segment.strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _try_braced_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
