from __future__ import annotations

import json

from DASH import setup


class _FakeApp:
    def __init__(self, repo_root) -> None:
        self.repo_root = repo_root
        self.lines: list[str] = []

    def _append_system_line(self, text: str) -> None:
        self.lines.append(text)


def test_setup_runtime_updates_active_profile(tmp_path):
    app = _FakeApp(tmp_path)
    agents_dir = tmp_path / "Agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "system_prompt.md").write_text("base prompt\n", encoding="utf-8")
    (agents_dir / "default.json").write_text(
        json.dumps(
            {
                "name": "default",
                "system_prompt_file": "system_prompt.md",
                "runtime": {"ui": "terminal", "provider": "openai", "model": "gpt-4o-mini"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    handled = setup.run("/setup runtime tui ollama llama3.1", app=app)

    assert handled is True
    profile = json.loads((agents_dir / "default.json").read_text(encoding="utf-8"))
    assert profile["runtime"] == {"ui": "tui", "provider": "ollama", "model": "llama3.1"}
