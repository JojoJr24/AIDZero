#!/usr/bin/env bash
# Script: run-tool-gateway.sh
# Objective: launch the MCP tool-search gateway implemented in Python.
# Notes: gateway MCP server config lives at MCP/mcporter.json.
# Usage: bash MCP/run-tool-gateway.sh
# Inputs: none. Outputs: gateway process running on stdio.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
GATEWAY_MAIN="$SCRIPT_DIR/tool-gateway/gateway_server.py"

if [[ ! -f "$GATEWAY_MAIN" ]]; then
  echo "Gateway server not found: $GATEWAY_MAIN" >&2
  exit 1
fi

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python 3 runtime missing in PATH and .venv/bin/python is unavailable." >&2
  exit 1
fi

exec "$PYTHON_BIN" "$GATEWAY_MAIN" stdio
