"""Tests for generated agent provider runtime helpers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.generated_agent_runtime import create_provider_from_config


def test_create_provider_from_config_finds_non_generic_provider_class(tmp_path: Path) -> None:
    project_root = tmp_path
    provider_dir = project_root / "LLMProviders" / "AID-openai"
    provider_dir.mkdir(parents=True)
    (provider_dir / "provider.py").write_text(
        (
            "class ProviderError(Exception):\n"
            "    pass\n\n"
            "class OpenAIProvider:\n"
            "    def __init__(self, api_key=None):\n"
            "        self.api_key = api_key\n"
            "    def generate_text(self, model, prompt, **kwargs):\n"
            "        return 'ok'\n"
            "    def list_model_names(self, *, page_size=100):\n"
            "        return ['demo']\n"
        ),
        encoding="utf-8",
    )

    provider = create_provider_from_config(
        project_root=project_root,
        config={
            "provider": "AID-openai",
            "provider_options": {"api_key": "abc"},
        },
    )

    assert provider.__class__.__name__ == "OpenAIProvider"
