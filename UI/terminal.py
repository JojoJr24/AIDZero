"""Terminal UI module loaded dynamically from UI/*.py."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.engine import AgentEngine
from agent.gateway import TriggerGateway
from agent.llm_client import LLMClient
from agent.memory import MemoryStore
from agent.prompt_history import PromptHistoryStore
from agent.storage import JsonlStore
from agent.terminal_app import TerminalApp
from agent.tooling import build_default_tool_registry


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
    trigger = options.get("trigger", "interactive").strip().lower() or "interactive"

    llm = LLMClient(repo_root=root, provider_name=provider_name, model=model)
    memory = MemoryStore(root / ".aidzero" / "memory.json")
    tools = build_default_tool_registry(root, memory)
    history_store = JsonlStore(root / ".aidzero" / "store" / "history.jsonl")
    output_store = JsonlStore(root / ".aidzero" / "store" / "output.jsonl")

    engine = AgentEngine(
        repo_root=root,
        llm=llm,
        tools=tools,
        history_store=history_store,
        memory_store=memory,
        output_store=output_store,
    )

    app = TerminalApp(
        repo_root=root,
        engine=engine,
        gateway=TriggerGateway(root),
        history=PromptHistoryStore(root),
    )
    return app.run(request=user_request, trigger=trigger)
