"""LLM-driven generator for the created agent's main.py entrypoint."""

from __future__ import annotations

from typing import Any

from .models import AgentPlan


class AgentEntrypointWriter:
    """Uses the configured LLM to generate main.py for the new agent."""

    def __init__(self, provider: Any, model: str, *, temperature: float = 0.2) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature

    def generate_main_py(self, *, user_request: str, plan: AgentPlan) -> str:
        prompt = self._build_prompt(user_request=user_request, plan=plan)
        raw = self.provider.generate_text(
            model=self.model,
            prompt=prompt,
            generation_config={"temperature": self.temperature},
        )
        code = _extract_python_code(raw)
        if not code.strip():
            raise ValueError("LLM returned empty main.py content.")
        normalized_code = code.rstrip() + "\n"
        _ensure_valid_python(normalized_code)
        return normalized_code

    @staticmethod
    def _build_prompt(*, user_request: str, plan: AgentPlan) -> str:
        return (
            "You are generating the entire content of main.py for a newly created Python agent.\n"
            "Return ONLY Python code, no markdown fences, no extra commentary.\n\n"
            "Context:\n"
            f"- agent_name: {plan.agent_name}\n"
            f"- project_folder: {plan.project_folder}\n"
            f"- goal: {plan.goal}\n"
            f"- summary: {plan.summary}\n"
            f"- required_llm_providers: {plan.required_llm_providers}\n"
            f"- required_skills: {plan.required_skills}\n"
            f"- required_tools: {plan.required_tools}\n"
            f"- required_mcp: {plan.required_mcp}\n"
            f"- required_ui: {plan.required_ui}\n"
            f"- implementation_steps: {plan.implementation_steps}\n"
            f"- original_user_request: {user_request.strip()}\n\n"
            "Requirements for main.py:\n"
            "1. Must be executable as the project's entrypoint.\n"
            "2. Must support non-interactive execution suitable for cron.\n"
            "3. Must support producing output that can be forwarded to chat systems.\n"
            "4. Include CLI arguments to control mode/output behavior.\n"
            "5. Use only Python standard library.\n"
            "6. Keep code production-friendly: clear functions, error handling, and __main__ guard.\n"
        )


def _extract_python_code(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if "```" not in text:
        return text

    segments = text.split("```")
    for segment in segments:
        candidate = segment.strip()
        if not candidate:
            continue
        if candidate.startswith("python"):
            candidate = candidate[len("python") :].lstrip()
        if "import " in candidate or "def " in candidate or "if __name__" in candidate:
            return candidate
    return text.replace("```", "").strip()


def _ensure_valid_python(source: str) -> None:
    try:
        compile(source, filename="generated_main.py", mode="exec")
    except SyntaxError as error:
        raise ValueError(
            f"LLM generated invalid Python for main.py: {error.msg} "
            f"(line {error.lineno}, column {error.offset})"
        ) from error
