"""TUI application wrapper."""

from __future__ import annotations

from pathlib import Path

from CORE.agents import AgentProfile, AgentProfileManager
from CORE.engine import AgentEngine
from CORE.gateway import TriggerGateway
from CORE.prompt_history import PromptHistoryStore
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
        del trigger
        return run_textual_tui(
            repo_root=self.repo_root,
            engine=self.engine,
            gateway=self.gateway,
            history=self.history,
            agent_manager=self.agent_manager,
            agent_profile=self.agent_profile,
            request=request,
            trigger="interactive",
        )
