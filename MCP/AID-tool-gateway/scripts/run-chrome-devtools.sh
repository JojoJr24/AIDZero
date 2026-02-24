#!/usr/bin/env bash
# Script: tool-gateway/scripts/run-chrome-devtools.sh
# Objective: launch the chrome-devtools MCP server using the locally installed package to avoid npx prompts.
# Usage: MCP/AID-tool-gateway/scripts/run-chrome-devtools.sh [chrome-devtools args]
# Inputs: arguments are forwarded verbatim to chrome-devtools-mcp.
# Side-effects: requires Node.js in PATH and npm install completed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js runtime missing in PATH. Install Node.js or set PATH accordingly." >&2
  exit 1
fi

SERVER_BIN="$ROOT/node_modules/.bin/chrome-devtools-mcp"

if [[ ! -x "$SERVER_BIN" ]]; then
  echo "chrome-devtools-mcp binary missing. Run npm install inside $ROOT first." >&2
  exit 1
fi

exec "$SERVER_BIN" "$@"
