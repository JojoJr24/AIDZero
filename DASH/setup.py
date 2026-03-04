"""Slash command to configure trigger integrations."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

DASH_COMMANDS = [
    {"command": "/setup show", "description": "Show trigger integration config"},
    {"command": "/setup cron <schedule>", "description": "Configure cron schedule and files"},
    {"command": "/setup heartbeat <schedule>", "description": "Configure heartbeat schedule and files"},
    {"command": "/setup message-origin <name> [path]", "description": "Register message origin"},
    {"command": "/setup webhook-origin <name> [path]", "description": "Register webhook origin"},
]


def match(raw: str) -> bool:
    return raw.startswith("/setup")


def run(raw: str, *, app: Any) -> bool:
    repo_root = getattr(app, "repo_root", None)
    if not isinstance(repo_root, Path):
        app._append_system_line("Cannot resolve repository root for /setup.")
        return True

    args = _parse_args(raw)
    if not args:
        app._append_system_line(_usage())
        return True

    config_path = repo_root / ".aidzero" / "trigger_sources.json"
    config = _read_json(config_path)

    action = args[0]
    if action == "show":
        _show_config(config, app=app)
        return True

    if action in {"cron", "heartbeat"}:
        if len(args) < 2:
            app._append_system_line(_usage())
            return True
        schedule = " ".join(args[1:]).strip()
        if not schedule:
            app._append_system_line(_usage())
            return True
        config[f"{action}_schedule"] = schedule
        if action == "cron":
            config.setdefault("cron_path", ".aidzero/cron_prompt.txt")
        else:
            config.setdefault("heartbeat_path", "HEARTBEAT.md")
        _write_json(config_path, config)
        _generate_runner_scripts(repo_root, app=app)
        _generate_crontab(repo_root, config, app=app)
        return True

    if action in {"message-origin", "webhook-origin"}:
        if len(args) < 2:
            app._append_system_line(_usage())
            return True
        origin_name = args[1].strip().lower()
        if not origin_name:
            app._append_system_line("Origin name cannot be empty.")
            return True
        kind = "message_origins" if action == "message-origin" else "webhook_origins"
        default_basename = "messages" if action == "message-origin" else "webhooks"
        if len(args) >= 3:
            origin_path = " ".join(args[2:]).strip()
        else:
            origin_path = f".aidzero/inbox/{default_basename}_{origin_name}.jsonl"
        _upsert_origin(config, kind=kind, origin_name=origin_name, origin_path=origin_path)
        _write_json(config_path, config)
        _ensure_parent_and_file(repo_root, origin_path)
        app._append_system_line(f"Registered {action} '{origin_name}' -> {origin_path}")
        return True

    app._append_system_line(_usage())
    return True


def _parse_args(raw: str) -> list[str]:
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return []
    if not tokens or tokens[0] != "/setup":
        return []
    return [token.strip() for token in tokens[1:] if token.strip()]


def _usage() -> str:
    return (
        "Usage: /setup show | /setup cron <schedule> | /setup heartbeat <schedule> | "
        "/setup message-origin <name> [path] | /setup webhook-origin <name> [path]"
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _show_config(config: dict[str, Any], *, app: Any) -> None:
    if not config:
        app._append_system_line("No setup yet. Use /setup cron ... or /setup message-origin ...")
        return
    app._append_system_line("Current trigger setup:")
    for key in sorted(config):
        app._append_system_line(f"- {key}: {config[key]}")


def _upsert_origin(config: dict[str, Any], *, kind: str, origin_name: str, origin_path: str) -> None:
    rows = config.get(kind)
    if not isinstance(rows, list):
        rows = []
    filtered = [row for row in rows if isinstance(row, dict) and row.get("name") != origin_name]
    filtered.append({"name": origin_name, "path": origin_path})
    config[kind] = filtered


def _ensure_parent_and_file(repo_root: Path, raw_path: str) -> None:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def _generate_runner_scripts(repo_root: Path, *, app: Any) -> None:
    runtime_config = _read_json(repo_root / ".aidzero" / "runtime_config.json")
    provider = str(runtime_config.get("provider", "")).strip() or "YOUR_PROVIDER"
    model = str(runtime_config.get("model", "")).strip() or "YOUR_MODEL"
    ui = str(runtime_config.get("ui", "")).strip() or "terminal"
    cmd_base = (
        f"uv run AIDZero.py --provider {provider} --model {model} --ui {ui}"
    )

    scripts_dir = repo_root / ".aidzero" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    cron_script = scripts_dir / "run_cron.sh"
    cron_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(repo_root))}",
                f"{cmd_base} --trigger cron --request __cron_tick__",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cron_script.chmod(0o755)

    heartbeat_script = scripts_dir / "run_heartbeat.sh"
    heartbeat_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(repo_root))}",
                f"{cmd_base} --trigger heartbeat --request __heartbeat_tick__",
                "",
            ]
        ),
        encoding="utf-8",
    )
    heartbeat_script.chmod(0o755)

    app._append_system_line("Generated runner scripts in .aidzero/scripts/")


def _generate_crontab(repo_root: Path, config: dict[str, Any], *, app: Any) -> None:
    cron_schedule = str(config.get("cron_schedule", "")).strip()
    heartbeat_schedule = str(config.get("heartbeat_schedule", "")).strip()

    lines = [
        "# AIDZero managed entries",
    ]
    if cron_schedule:
        lines.append(
            f"{cron_schedule} {shlex.quote(str(repo_root / '.aidzero' / 'scripts' / 'run_cron.sh'))} "
            f">> {shlex.quote(str(repo_root / '.aidzero' / 'output' / 'cron.log'))} 2>&1"
        )
    if heartbeat_schedule:
        lines.append(
            f"{heartbeat_schedule} {shlex.quote(str(repo_root / '.aidzero' / 'scripts' / 'run_heartbeat.sh'))} "
            f">> {shlex.quote(str(repo_root / '.aidzero' / 'output' / 'heartbeat.log'))} 2>&1"
        )

    setup_dir = repo_root / ".aidzero" / "setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / ".aidzero" / "output").mkdir(parents=True, exist_ok=True)

    crontab_file = setup_dir / "aidzero.crontab"
    crontab_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    install_script = setup_dir / "install_cron.sh"
    install_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"crontab {shlex.quote(str(crontab_file))}",
                'echo "Installed AIDZero cron entries."',
                "",
            ]
        ),
        encoding="utf-8",
    )
    install_script.chmod(0o755)

    app._append_system_line("Generated .aidzero/setup/aidzero.crontab and install_cron.sh")
