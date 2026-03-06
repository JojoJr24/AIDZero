"""Launcher that starts core API + UI together as separate processes."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from CORE.agents import AgentProfileManager
from CORE.api_server import serve_core_api
from CORE.ui_registry import UIRegistry


def _parse_ui_options(raw_items: list[str], trigger: str) -> dict[str, str]:
    options: dict[str, str] = {"trigger": trigger}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --ui-option '{item}'. Use KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --ui-option '{item}'. KEY cannot be empty.")
        options[key] = value.strip()
    return options


def _parse_args() -> Any:
    parser = ArgumentParser(description="Run core API and UI layers together")
    parser.add_argument("--ui", default=None, help="UI name from UI/<name>/entrypoint.py")
    parser.add_argument("--host", default="0.0.0.0", help="Core bind host/IP")
    parser.add_argument("--port", type=int, default=8765, help="Core bind port")
    parser.add_argument("--request", help="Optional one-shot prompt")
    parser.add_argument("--agent", default=None, help="Agent profile name from Agents/*.json")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for core health endpoint",
    )
    parser.add_argument(
        "--ui-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra option passed to UI entrypoint (repeatable)",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root path",
    )
    return parser.parse_args()


def _health_ready(core_url: str, *, timeout_seconds: float) -> bool:
    health_url = core_url.rstrip("/") + "/health"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(health_url, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict) and bool(payload.get("ok", False)):
                return True
        except (OSError, ValueError, URLError):
            time.sleep(0.2)
    return False


def _connect_host(host: str) -> str:
    raw = host.strip()
    if raw in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return raw


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()

    profile_manager = AgentProfileManager(repo_root)
    if args.agent and str(args.agent).strip():
        try:
            active_profile = profile_manager.set_active_profile(str(args.agent).strip())
        except Exception as error:  # noqa: BLE001
            print(f"error> {error}")
            return 2
    else:
        active_profile = profile_manager.get_active_profile()

    ui_name = (str(args.ui).strip() if args.ui else active_profile.runtime_ui)
    provider_name = active_profile.runtime_provider
    model = active_profile.runtime_model

    ui_registry = UIRegistry(repo_root)
    if ui_name not in ui_registry.names():
        print(f"error> unknown ui '{ui_name}'")
        return 2
    ui_type = ui_registry.ui_type(ui_name)

    try:
        ui_options = _parse_ui_options(list(args.ui_option), trigger="interactive")
    except ValueError as error:
        print(f"error> {error}")
        return 2

    bind_host = str(args.host).strip()
    bind_port = int(args.port)
    core_url = f"http://{_connect_host(bind_host)}:{bind_port}"
    ui_options["core_url"] = core_url

    cmd = [
        sys.executable,
        "-m",
        "core.api_server",
        "--agent",
        active_profile.name,
        "--host",
        bind_host,
        "--port",
        str(bind_port),
        "--repo-root",
        str(repo_root),
    ]

    print("Starting split runtime:")
    print(f"- core: {bind_host}:{bind_port} ({provider_name} / {model})")
    print(f"- ui: {ui_name}")

    if ui_type == "thirdparty":
        if args.request:
            print("warning> --request is ignored for thirdparty UI mode.")
        print("info> thirdparty UI selected; running only core API for LAN clients.")
        try:
            return serve_core_api(
                repo_root=repo_root,
                provider_name=provider_name,
                model=model,
                host=bind_host,
                port=bind_port,
            )
        except RuntimeError as error:
            print(f"error> {error}")
            print("tip> Cerrá el proceso que usa ese puerto o ejecutá con --port 8766 (u otro libre).")
            return 2

    core_proc = subprocess.Popen(cmd)
    try:
        if not _health_ready(core_url, timeout_seconds=float(args.startup_timeout)):
            if core_proc.poll() is not None:
                print("error> El proceso core API terminó antes de estar listo.")
                print("tip> Revisá el error mostrado arriba o probá otro puerto con --port 8766.")
                return 2
            print("error> core API did not become healthy in time.")
            return 2
        return ui_registry.run(
            ui_name,
            provider_name=provider_name,
            model=model,
            user_request=args.request,
            repo_root=repo_root,
            ui_options=ui_options,
        )
    finally:
        if core_proc.poll() is None:
            core_proc.terminate()
            try:
                core_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                core_proc.kill()
                core_proc.wait(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
