#!/usr/bin/env bash
# Script: run-tool-gateway.sh
# Objective: launch the MCP tool-search gateway via npm start.
# Usage: bash run-tool-gateway.sh
# Inputs: none. Outputs: gateway process running on stdio.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/MCP/AID-tool-gateway"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js runtime missing in PATH. Install Node.js or set PATH accordingly." >&2
  exit 1
fi

cd "$ROOT"
exec npm start
