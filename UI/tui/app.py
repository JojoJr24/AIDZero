"""TUI application wrapper."""

from __future__ import annotations

from pathlib import Path

from core.agents import AgentProfile, AgentProfileManager
from core.engine import AgentEngine
from core.gateway import TriggerGateway
from core.prompt_history import PromptHistoryStore
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
        agent_manager: AgentProfileManager,
        agent_profile: AgentProfile,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.engine = engine
        self.gateway = gateway
        self.history = history
        self.agent_manager = agent_manager
        self.agent_profile = agent_profile

    def run(self, *, request: str | None, trigger: str) -> int:
        active_trigger = trigger.strip().lower() or "interactive"
        return run_textual_tui(
            repo_root=self.repo_root,
            engine=self.engine,
            gateway=self.gateway,
            history=self.history,
            agent_manager=self.agent_manager,
            agent_profile=self.agent_profile,
            request=request,
            trigger=active_trigger,
        )
