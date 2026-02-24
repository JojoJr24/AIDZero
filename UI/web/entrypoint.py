"""Web UI runtime entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UI.web.agent_web import run_web_agent


def _option_as_str(options: dict[str, str], key: str, default: str) -> str:
    raw = options.get(key)
    if raw is None:
        return default
    value = str(raw).strip()
    return value or default


def _option_as_int(options: dict[str, str], key: str, default: int) -> int:
    raw = options.get(key)
    if raw is None:
        return default
    value = str(raw).strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Invalid UI option '{key}': expected integer, got '{value}'.") from error


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
    options = ui_options or {}
    host = _option_as_str(options, "host", "127.0.0.1")
    port = _option_as_int(options, "port", 8787)
    return run_web_agent(
        provider_name=provider_name,
        model=model,
        user_request=user_request,
        dry_run=dry_run,
        overwrite=overwrite,
        yes=yes,
        repo_root=repo_root,
        host=host,
        port=port,
    )
