"""Tests for LLM-driven child main.py generation and repair loop."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.entrypoint_writer import AgentEntrypointWriter
from agent.models import AgentPlan


class FakeProvider:
    """Provider stub with deterministic generation responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_text(self, model: str, prompt: str, **kwargs: object) -> str:
        self.calls.append({"model": model, "prompt": prompt, "kwargs": kwargs})
        if not self._responses:
            raise AssertionError("FakeProvider has no more prepared responses.")
        return self._responses.pop(0)


def _build_plan() -> AgentPlan:
    return AgentPlan(
        agent_name="demo_agent",
        project_folder="generated_agents/demo_agent",
        goal="Build a demo agent.",
        summary="Demo summary.",
        required_llm_providers=["AID-claude"],
        required_skills=[],
        required_tools=[],
        required_mcp=[],
        required_ui=["terminal"],
        folder_blueprint=["src", "tests"],
        implementation_steps=["Create child main entrypoint."],
    )


def test_generate_main_py_repairs_invalid_python() -> None:
    provider = FakeProvider(
        responses=[
            "def broken(:\n    pass\n",
            (
                "import argparse\n\n"
                "def main() -> int:\n"
                "    parser = argparse.ArgumentParser()\n"
                "    parser.add_argument('--name', default='world')\n"
                "    args = parser.parse_args()\n"
                "    print(f'hello {args.name}')\n"
                "    return 0\n\n"
                "if __name__ == '__main__':\n"
                "    raise SystemExit(main())\n"
            ),
        ]
    )
    writer = AgentEntrypointWriter(provider=provider, model="test-model", max_repair_attempts=2)

    generated = writer.generate_main_py(user_request="create greeting agent", plan=_build_plan())

    assert "def main() -> int:" in generated
    assert len(provider.calls) == 2
    first_prompt = str(provider.calls[0]["prompt"])
    assert "NEW child agent" in first_prompt
    assert "original_user_request: create greeting agent" in first_prompt
    second_prompt = str(provider.calls[1]["prompt"])
    assert "Validation error:" in second_prompt
    assert "invalid syntax" in second_prompt


def test_generate_main_py_repairs_when_contract_is_missing() -> None:
    provider = FakeProvider(
        responses=[
            "def execute():\n    return 1\n",
            (
                "import argparse\n\n"
                "def main() -> int:\n"
                "    parser = argparse.ArgumentParser()\n"
                "    parser.add_argument('--text', required=True)\n"
                "    args = parser.parse_args()\n"
                "    print(args.text.upper())\n"
                "    return 0\n\n"
                "if __name__ == '__main__':\n"
                "    raise SystemExit(main())\n"
            ),
        ]
    )
    writer = AgentEntrypointWriter(provider=provider, model="test-model", max_repair_attempts=2)

    generated = writer.generate_main_py(user_request="uppercase text", plan=_build_plan())

    assert "argparse" in generated
    assert len(provider.calls) == 2
    assert "must define a main()" in str(provider.calls[1]["prompt"])


def test_generate_main_py_raises_after_repair_attempts() -> None:
    provider = FakeProvider(
        responses=[
            "if True print('one')\n",
            "if True print('two')\n",
        ]
    )
    writer = AgentEntrypointWriter(provider=provider, model="test-model", max_repair_attempts=1)

    with pytest.raises(ValueError) as exc_info:
        writer.generate_main_py(user_request="bad code", plan=_build_plan())

    message = str(exc_info.value)
    assert "after 2 attempts" in message
    assert "invalid syntax" in message
    assert len(provider.calls) == 2


def test_generate_main_py_rejects_empty_request() -> None:
    writer = AgentEntrypointWriter(provider=FakeProvider(["ignored"]), model="test-model")
    with pytest.raises(ValueError):
        writer.generate_main_py(user_request="   ", plan=_build_plan())
