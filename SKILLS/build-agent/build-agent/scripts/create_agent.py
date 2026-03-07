#!/usr/bin/env python3
"""Legacy placeholder. Runtime agents must be created with agent-creator."""

from __future__ import annotations


def main() -> int:
    print("Deprecated: use SKILLS/agent-creator for runtime agent profiles.")
    print("Expected format: Agents/<name>/<name>.json + system_prompt.md + HeadlessPrompt.txt")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
