"""Desktop/browser automation actions."""

from __future__ import annotations

from datetime import UTC, datetime
import os
import shutil
import sys
from typing import Any

from TOOLS._helpers import (
    normalize_timeout,
    require_binary,
    require_desktop_session,
    run_command,
    safe_resolve,
)

TOOL_NAME = "computer_control"
TOOL_DESCRIPTION = "Control desktop/browser actions (open_url, type, keys, mouse, screenshot, run)."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "url": {"type": "string"},
        "text": {"type": "string"},
        "key": {"type": "string"},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "button": {"type": "string"},
        "clicks": {"type": "integer", "minimum": 1, "maximum": 20},
        "delay_ms": {"type": "integer", "minimum": 0, "maximum": 2000},
        "path": {"type": "string"},
        "command": {"type": "string"},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 180},
        "target": {"type": "string"},
        "payload": {},
    },
    "required": ["action"],
    "additionalProperties": True,
}


def run(arguments: dict[str, Any], *, repo_root, memory):
    del memory
    action = str(arguments.get("action", "")).strip().lower()
    if not action:
        raise ValueError("'action' is required.")

    if action == "run":
        return _computer_run(repo_root, arguments)
    if action == "open_url":
        return _computer_open_url(arguments)
    if action == "type_text":
        return _computer_type_text(arguments)
    if action == "key_press":
        return _computer_key_press(arguments)
    if action == "move_mouse":
        return _computer_move_mouse(arguments)
    if action == "mouse_click":
        return _computer_mouse_click(arguments)
    if action == "screenshot":
        return _computer_screenshot(repo_root, arguments)

    raise ValueError(
        "Unsupported computer_control action. "
        "Use one of: run, open_url, type_text, key_press, move_mouse, mouse_click, screenshot."
    )


def _computer_run(repo_root, arguments: dict[str, Any]) -> dict[str, Any]:
    command = str(arguments.get("command", "")).strip()
    if not command:
        raise ValueError("'command' is required for action=run.")

    timeout = normalize_timeout(arguments, default=30, maximum=180)
    import subprocess

    result = subprocess.run(
        command,
        cwd=repo_root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "status": "ok",
        "action": "run",
        "exit_code": result.returncode,
        "stdout": result.stdout[-6000:],
        "stderr": result.stderr[-6000:],
    }


def _computer_open_url(arguments: dict[str, Any]) -> dict[str, Any]:
    url = str(arguments.get("url") or arguments.get("target") or "").strip()
    if not url:
        raise ValueError("'url' (or 'target') is required for action=open_url.")

    timeout = normalize_timeout(arguments, default=15, maximum=60)
    if sys.platform.startswith("linux"):
        binary = require_binary("xdg-open")
        run_result = run_command([binary, url], timeout_seconds=timeout)
    elif sys.platform == "darwin":
        binary = require_binary("open")
        run_result = run_command([binary, url], timeout_seconds=timeout)
    elif sys.platform.startswith("win") and hasattr(os, "startfile"):
        os.startfile(url)  # type: ignore[attr-defined]
        run_result = {"command": ["startfile", url], "exit_code": 0, "stdout": "", "stderr": ""}
    else:
        raise RuntimeError(f"Unsupported platform for open_url: {sys.platform}")

    return {
        "status": "ok" if int(run_result["exit_code"]) == 0 else "error",
        "action": "open_url",
        "url": url,
        "result": run_result,
    }


def _computer_type_text(arguments: dict[str, Any]) -> dict[str, Any]:
    require_desktop_session()
    binary = require_binary("xdotool")
    text = str(arguments.get("text") or arguments.get("target") or "").strip()
    if not text:
        raise ValueError("'text' (or 'target') is required for action=type_text.")

    delay_ms = int(arguments.get("delay_ms", 12))
    run_result = run_command([binary, "type", "--delay", str(delay_ms), "--", text], timeout_seconds=20)
    return {
        "status": "ok" if int(run_result["exit_code"]) == 0 else "error",
        "action": "type_text",
        "text_length": len(text),
        "result": run_result,
    }


def _computer_key_press(arguments: dict[str, Any]) -> dict[str, Any]:
    require_desktop_session()
    binary = require_binary("xdotool")
    key = str(arguments.get("key") or arguments.get("target") or "").strip()
    if not key:
        raise ValueError("'key' (or 'target') is required for action=key_press.")

    run_result = run_command([binary, "key", "--clearmodifiers", key], timeout_seconds=20)
    return {
        "status": "ok" if int(run_result["exit_code"]) == 0 else "error",
        "action": "key_press",
        "key": key,
        "result": run_result,
    }


def _computer_move_mouse(arguments: dict[str, Any]) -> dict[str, Any]:
    require_desktop_session()
    binary = require_binary("xdotool")
    if "x" not in arguments or "y" not in arguments:
        raise ValueError("'x' and 'y' are required for action=move_mouse.")

    x = int(arguments["x"])
    y = int(arguments["y"])
    run_result = run_command([binary, "mousemove", str(x), str(y)], timeout_seconds=20)
    return {
        "status": "ok" if int(run_result["exit_code"]) == 0 else "error",
        "action": "move_mouse",
        "x": x,
        "y": y,
        "result": run_result,
    }


def _computer_mouse_click(arguments: dict[str, Any]) -> dict[str, Any]:
    require_desktop_session()
    binary = require_binary("xdotool")

    button_raw = str(arguments.get("button", "left")).strip().lower()
    button_map = {"left": "1", "middle": "2", "right": "3"}
    button = button_map.get(button_raw, button_raw)
    clicks = int(arguments.get("clicks", 1))

    if "x" in arguments and "y" in arguments:
        x = int(arguments["x"])
        y = int(arguments["y"])
        move_result = run_command([binary, "mousemove", str(x), str(y)], timeout_seconds=20)
        if int(move_result["exit_code"]) != 0:
            return {"status": "error", "action": "mouse_click", "result": move_result}

    run_result = run_command(
        [binary, "click", "--repeat", str(clicks), button],
        timeout_seconds=20,
    )
    return {
        "status": "ok" if int(run_result["exit_code"]) == 0 else "error",
        "action": "mouse_click",
        "button": button_raw,
        "clicks": clicks,
        "result": run_result,
    }


def _computer_screenshot(repo_root, arguments: dict[str, Any]) -> dict[str, Any]:
    require_desktop_session()

    raw_path = str(arguments.get("path", "")).strip()
    if raw_path:
        target = safe_resolve(repo_root, raw_path)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        target = safe_resolve(repo_root, f".aidzero/output/screenshot-{stamp}.png")
    target.parent.mkdir(parents=True, exist_ok=True)

    backend_command: list[str] | None = None
    for candidate in ("import", "scrot", "gnome-screenshot", "grim"):
        binary = shutil.which(candidate)
        if not binary:
            continue
        if candidate == "import":
            backend_command = [binary, "-window", "root", str(target)]
        elif candidate == "scrot":
            backend_command = [binary, str(target)]
        elif candidate == "gnome-screenshot":
            backend_command = [binary, "-f", str(target)]
        elif candidate == "grim":
            backend_command = [binary, str(target)]
        if backend_command:
            break

    if backend_command is None:
        raise RuntimeError("No screenshot backend found (import/scrot/gnome-screenshot/grim).")

    run_result = run_command(backend_command, timeout_seconds=30)
    return {
        "status": "ok" if int(run_result["exit_code"]) == 0 and target.exists() else "error",
        "action": "screenshot",
        "path": str(target.relative_to(repo_root)),
        "result": run_result,
    }
