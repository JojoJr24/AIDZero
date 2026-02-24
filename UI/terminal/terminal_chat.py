#!/usr/bin/env python3
"""Simple terminal chat UI for OpenAI-compatible LLM endpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.ui_display import to_ui_model_label


@dataclass
class ChatConfig:
    base_url: str
    api_key: str
    model: str
    system_prompt: str | None
    temperature: float
    max_tokens: int
    timeout: int


def parse_args() -> ChatConfig:
    parser = argparse.ArgumentParser(
        description="Terminal chat UI for OpenAI-compatible endpoints.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")),
        help="Base URL for the LLM API (default: %(default)s).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "not-needed")),
        help="API key for the provider (default: env or 'not-needed').",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        help="Model name to use (default: %(default)s).",
    )
    parser.add_argument(
        "--system-prompt",
        default=os.getenv("LLM_SYSTEM_PROMPT"),
        help="Optional system prompt for the session.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens in each response (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds (default: %(default)s).",
    )
    args = parser.parse_args()
    return ChatConfig(
        base_url=args.base_url.rstrip("/"),
        api_key=args.api_key,
        model=args.model,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )


def extract_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Invalid response: missing 'choices'.")

    message = choices[0].get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_chunks = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                text_chunks.append(part["text"])
        if text_chunks:
            return "\n".join(text_chunks).strip()

    raise ValueError("Invalid response: unsupported content format.")


def call_chat_completion(config: ChatConfig, messages: list[dict[str, str]]) -> str:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{config.base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    request = urllib.request.Request(url=url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {error_body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Connection error: {error.reason}") from error

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON response: {body[:400]}") from error

    return extract_text(parsed)


def print_help() -> None:
    print("Commands:")
    print("  /help   Show this help")
    print("  /reset  Clear conversation history")
    print("  /exit   Quit the chat")


def run_chat(config: ChatConfig) -> int:
    messages: list[dict[str, str]] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})

    print("Terminal Chat UI")
    print(f"Endpoint: {config.base_url}/chat/completions")
    print(f"Model: {to_ui_model_label(config.model)}")
    print("Type /help for commands.")

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            return 0

        if not user_input:
            continue

        if user_input in {"/exit", "/quit", "/q"}:
            print("Exiting chat.")
            return 0
        if user_input == "/help":
            print_help()
            continue
        if user_input == "/reset":
            messages = [{"role": "system", "content": config.system_prompt}] if config.system_prompt else []
            print("Conversation reset.")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            assistant_text = call_chat_completion(config, messages)
        except Exception as error:  # noqa: BLE001
            messages.pop()
            print(f"error> {error}")
            continue

        messages.append({"role": "assistant", "content": assistant_text})
        print(f"assistant> {assistant_text}")


def main() -> int:
    config = parse_args()
    if not config.model:
        print("error> Missing model name. Use --model or set LLM_MODEL.", file=sys.stderr)
        return 2
    return run_chat(config)


if __name__ == "__main__":
    raise SystemExit(main())
