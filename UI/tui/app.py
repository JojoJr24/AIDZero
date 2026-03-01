"""TUI application wrapper."""

from __future__ import annotations

from pathlib import Path

from agent.engine import AgentEngine
from agent.gateway import TriggerGateway
from agent.prompt_history import PromptHistoryStore
from UI.tui.textual_app import run_textual_tui


class TUIApp:
    """Textual-backed TUI wrapper."""

    def __init__(
        self,
        *,
        repo_root: Path,
        engine: AgentEngine,
        gateway: TriggerGateway,
        history: PromptHistoryStore,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.engine = engine
        self.gateway = gateway
        self.history = history

    def run(self, *, request: str | None, trigger: str) -> int:
        active_trigger = trigger.strip().lower() or "interactive"
        return run_textual_tui(
            repo_root=self.repo_root,
            engine=self.engine,
            gateway=self.gateway,
            history=self.history,
            request=request,
            trigger=active_trigger,
        )
