"""Tests for prompt history persistence."""

from __future__ import annotations

from pathlib import Path

from agent.prompt_history import PromptHistoryStore


def test_prompt_history_store_adds_and_orders_prompts(tmp_path: Path) -> None:
    store = PromptHistoryStore(tmp_path, max_prompts=3)
    store.add_prompt("first prompt")
    store.add_prompt("second prompt")
    store.add_prompt("third prompt")

    assert store.list_prompts() == ["third prompt", "second prompt", "first prompt"]


def test_prompt_history_store_deduplicates_and_limits(tmp_path: Path) -> None:
    store = PromptHistoryStore(tmp_path, max_prompts=2)
    store.add_prompt("one")
    store.add_prompt("two")
    store.add_prompt("one")
    store.add_prompt("three")

    assert store.list_prompts() == ["three", "one"]


def test_prompt_history_store_ignores_blank_prompt(tmp_path: Path) -> None:
    store = PromptHistoryStore(tmp_path)
    store.add_prompt("real")
    store.add_prompt("   ")
    assert store.list_prompts() == ["real"]
