from __future__ import annotations

from CORE.prompt_history import PromptHistoryStore


def test_prompt_history_persists_and_deduplicates(tmp_path):
    store = PromptHistoryStore(tmp_path, max_items=3)

    store.add_prompt("one")
    store.add_prompt("two")
    store.add_prompt("one")
    store.add_prompt("three")
    store.add_prompt("four")

    assert store.list_prompts() == ["four", "three", "one"]

    store_reloaded = PromptHistoryStore(tmp_path, max_items=3)
    assert store_reloaded.list_prompts() == ["four", "three", "one"]
