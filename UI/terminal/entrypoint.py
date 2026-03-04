"""Terminal UI entrypoint module."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.ui_runtime import build_ui_runtime
from UI.terminal.app import TerminalApp


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

    runtime = build_ui_runtime(repo_root=root, provider_name=provider_name, model=model)
    app = TerminalApp(
        repo_root=root,
        engine=runtime.engine,
        gateway=runtime.gateway,
        history=runtime.history,
    )
    return app.run(request=user_request, trigger=trigger)
