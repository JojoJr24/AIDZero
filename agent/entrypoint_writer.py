"""LLM-driven generator for child-agent main.py."""

from __future__ import annotations

from typing import Any

from .models import AgentPlan


class AgentEntrypointWriter:
    """Generates a task-specific main.py for the child agent using the configured model."""

    def __init__(
        self,
        provider: Any,
        model: str,
        *,
        temperature: float = 0.2,
        max_repair_attempts: int = 2,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_repair_attempts = max(0, max_repair_attempts)

    def generate_main_py(self, *, user_request: str, plan: AgentPlan) -> str:
        request = user_request.strip()
        if not request:
            raise ValueError("user_request cannot be empty.")

        base_prompt = self._build_prompt(user_request=request, plan=plan)
        prompt = base_prompt
        total_attempts = self.max_repair_attempts + 1
        last_error_message = ""

        for attempt in range(1, total_attempts + 1):
            raw = self.provider.generate_text(
                model=self.model,
                prompt=prompt,
                generation_config={"temperature": self.temperature},
            )
            code = _extract_python_code(raw)
            if not code.strip():
                last_error_message = "LLM returned empty main.py content."
            else:
                normalized_code = code.rstrip() + "\n"
                syntax_error = _get_syntax_error(normalized_code)
                if syntax_error is None:
                    contract_error = _validate_generated_contract(normalized_code)
                    if contract_error is None:
                        return normalized_code
                    last_error_message = contract_error
                else:
                    last_error_message = _format_syntax_error(syntax_error)

            if attempt == total_attempts:
                break
            prompt = self._build_repair_prompt(
                base_prompt=base_prompt,
                invalid_code=code,
                validation_error=last_error_message,
            )

        raise ValueError(
            "LLM generated invalid Python for child main.py after "
            f"{total_attempts} attempts: {last_error_message}"
        )

    @staticmethod
    def _build_prompt(*, user_request: str, plan: AgentPlan) -> str:
        return (
            "You are generating the full content of main.py for a NEW child agent.\n"
            "Return ONLY Python code, without markdown fences or commentary.\n\n"
            "The child agent MUST be purpose-built for the user request (not a generic template).\n"
            "Design the CLI inputs/outputs specifically for the requested behavior.\n\n"
            "Context from parent planner:\n"
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
            f"- original_user_request: {user_request}\n\n"
            "Runtime constraints:\n"
            "1. Write production-ready Python with clear functions and robust error handling.\n"
            "2. Include argparse-based CLI tailored to the request (inputs/options/output format).\n"
            "3. Print the final result to stdout in a useful structure for automation.\n"
            "4. Add an explicit __main__ guard and exit code handling.\n"
            "5. Use only Python standard library plus local project modules.\n"
            "6. Do not import parent-repository-only modules; imports must resolve in child workspace.\n"
            "7. LLM integration must go through the local LLMProviders layer only.\n"
            "8. Never call provider SDKs/APIs directly from main.py (no direct OpenAI/Anthropic/Gemini clients).\n"
            "9. If the requested behavior needs an LLM call, implement this exact flow:\n"
            "   - import LLMProvider from LLMProviders.base\n"
            "   - import load_runtime_config and create_provider_from_config from agent.generated_agent_runtime\n"
            "   - load config from agent_config.json via load_runtime_config(project_root=...)\n"
            "   - create provider via create_provider_from_config(project_root=..., config=config)\n"
            "   - resolve model from CLI override first, else config['model']\n"
            "   - pass config['generation_config'] when calling provider.generate_text/chat methods\n"
            "10. If the request does not need LLM calls, do not add unnecessary provider logic.\n"
            "11. The output code must be complete and executable as main.py.\n"
        )

    @staticmethod
    def _build_repair_prompt(
        *,
        base_prompt: str,
        invalid_code: str,
        validation_error: str,
    ) -> str:
        return (
            f"{base_prompt}\n"
            "The previous output failed validation.\n"
            f"Validation error: {validation_error}\n"
            "Fix the code and return a complete replacement for main.py.\n"
            "Return ONLY Python code.\n\n"
            "Previous invalid main.py:\n"
            f"{invalid_code}\n"
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
        if "def " in candidate or "if __name__" in candidate or "import " in candidate:
            return candidate
    return text.replace("```", "").strip()


def _get_syntax_error(source: str) -> SyntaxError | None:
    try:
        compile(source, filename="generated_child_main.py", mode="exec")
    except SyntaxError as error:
        return error
    return None


def _format_syntax_error(error: SyntaxError) -> str:
    line = error.lineno if error.lineno is not None else "?"
    column = error.offset if error.offset is not None else "?"
    return f"{error.msg} (line {line}, column {column})"


def _validate_generated_contract(source: str) -> str | None:
    if "def main(" not in source:
        return "Generated main.py must define a main() function."
    if "if __name__" not in source:
        return "Generated main.py must include a __main__ guard."
    if "argparse" not in source:
        return "Generated main.py must expose a CLI via argparse."
    return None
