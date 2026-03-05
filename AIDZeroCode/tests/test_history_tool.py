from __future__ import annotations

import json

from TOOLS.history_get import run as history_get_run


def test_history_get_reads_latest_entries_and_filters(tmp_path):
    history_path = tmp_path / ".aidzero" / "store" / "history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"event": {"prompt": "first"}, "response": "one"},
        {"event": {"prompt": "second"}, "response": "two"},
        {"event": {"prompt": "error case"}, "response": "fail"},
    ]
    history_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    tail_payload = history_get_run({"limit": 2}, repo_root=tmp_path, memory=None)
    assert tail_payload["count"] == 2
    assert [item["event"]["prompt"] for item in tail_payload["entries"]] == ["second", "error case"]

    filtered = history_get_run({"query": "error"}, repo_root=tmp_path, memory=None)
    assert filtered["count"] == 1
    assert filtered["entries"][0]["event"]["prompt"] == "error case"
