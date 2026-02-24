#!/usr/bin/env python3
"""Compatibility wrapper for the AIDZero root entrypoint."""

from AIDZero import main


if __name__ == "__main__":
    raise SystemExit(main())
