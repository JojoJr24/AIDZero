from __future__ import annotations

import json

from TOOLS.list_skills import run as list_skills_run
from TOOLS.read_skill import run as read_skill_run
from TOOLS.run_skill_script import run as run_skill_script_run


def _build_skill(tmp_path):
    skill_root = tmp_path / "SKILLS" / "demo"
    (skill_root / "references").mkdir(parents=True, exist_ok=True)
    (skill_root / "scripts").mkdir(parents=True, exist_ok=True)

    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: demo\n"
        "description: demo skill\n"
        "---\n\n"
        "# Demo skill\n",
        encoding="utf-8",
    )
    (skill_root / "references" / "guide.md").write_text("guide text", encoding="utf-8")
    (skill_root / "scripts" / "echo.py").write_text(
        "import json, sys\nprint(json.dumps({'argv': sys.argv[1:]}))\n",
        encoding="utf-8",
    )
    return skill_root


def test_list_skills_and_read_skill(tmp_path):
    _build_skill(tmp_path)

    listed = list_skills_run({}, repo_root=tmp_path, memory=None)
    assert listed["skills"]
    assert listed["skills"][0]["name"] == "demo"

    read_payload = read_skill_run(
        {
            "skill_name": "demo",
            "include_references": True,
            "include_reference_contents": True,
        },
        repo_root=tmp_path,
        memory=None,
    )
    assert "Demo skill" in read_payload["skill_md"]
    assert read_payload["references"][0]["name"] == "guide.md"


def test_run_skill_script_executes_python_file(tmp_path):
    _build_skill(tmp_path)

    payload = run_skill_script_run(
        {
            "skill_name": "demo",
            "script": "echo.py",
            "args": ["a", "b"],
        },
        repo_root=tmp_path,
        memory=None,
    )

    assert payload["exit_code"] == 0
    assert json.loads(payload["stdout"].strip()) == {"argv": ["a", "b"]}
