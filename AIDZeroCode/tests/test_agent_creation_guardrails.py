from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_default_prompt_requires_reading_agent_creator_skill_first():
    text = (_repo_root() / "Agents" / "default" / "system_prompt.md").read_text(encoding="utf-8")

    assert "Required order for agent profile tasks:" in text
    assert '`read_skill` with `skill_name="agent-creator"`' in text
    assert "then edit files in `Agents/`" in text


def test_planner_prompt_requires_reading_agent_creator_skill_first():
    text = (_repo_root() / "Agents" / "planificador" / "system_prompt.md").read_text(encoding="utf-8")

    assert "Before editing `Agents/<name>/...`, first call:" in text
    assert '`read_skill` with `skill_name="agent-creator"`' in text
    assert "then edit files." in text


def test_agent_creator_skill_declares_source_of_truth():
    text = (_repo_root() / "SKILLS" / "agent-creator" / "SKILL.md").read_text(encoding="utf-8")

    assert "source of truth for agent creation/editing" in text
