"""TUI entrypoint module."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.api_client import (
    CoreAPIClient,
    RemoteAgentEngine,
    RemoteAgentProfileManager,
    RemotePromptHistoryStore,
    RemoteTriggerGateway,
)
from core.ui_runtime import build_ui_runtime
from UI.tui.app import TUIApp


def run_ui(
    *,
    provider_name: str,
    model: str,
    user_request: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    yes: bool = False,
    repo_root: Path | None = None,
    ui_options: dict[str, str] | None = None,
) -> int:
    del dry_run, overwrite, yes

    root = (repo_root or REPO_ROOT).resolve()
    options = ui_options or {}
    core_url = options.get("core_url", "").strip()

    if core_url:
        client = CoreAPIClient(core_url)
        engine = RemoteAgentEngine(client)
        gateway = RemoteTriggerGateway(client)
        history = RemotePromptHistoryStore(client)
        agent_manager = RemoteAgentProfileManager(client, repo_root=root)
        agent_profile = agent_manager.get_active_profile()
    else:
        runtime = build_ui_runtime(repo_root=root, provider_name=provider_name, model=model)
        engine = runtime.engine
        gateway = runtime.gateway
        history = runtime.history
        agent_manager = runtime.agent_manager
        agent_profile = runtime.agent_profile

    app = TUIApp(
        repo_root=root,
        engine=engine,
        gateway=gateway,
        history=history,
        agent_manager=agent_manager,
        agent_profile=agent_profile,
    )
    return app.run(request=user_request, trigger="interactive")
