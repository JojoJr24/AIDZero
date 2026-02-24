"""Terminal UI runtime entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UI.terminal.agent_terminal import run_terminal_agent


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
    del ui_options
    return run_terminal_agent(
        provider_name=provider_name,
        model=model,
        user_request=user_request,
        dry_run=dry_run,
        overwrite=overwrite,
        yes=yes,
        repo_root=repo_root,
    )
