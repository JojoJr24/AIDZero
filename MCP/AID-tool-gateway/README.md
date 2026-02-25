# AID Tool Gateway

MCP stdio gateway that exposes:

- `tool_search`
- `tool_describe`
- `tool_call`
- `tool_health`

## Runtime Config

Gateway MCP server definitions are loaded from:

```text
<repo-root>/MCP/mcporter.json
```

On startup, if legacy config exists at `.aidzero/mcporter.json` or `MCP/AID-tool-gateway/config/mcporter.json`, it is copied to `MCP/mcporter.json`.

## Add MCP Servers

1. Open `MCP/mcporter.json`.
2. Add a new entry under `mcpServers`.
3. Restart the gateway.

Example:

```json
{
  "mcpServers": {
    "deep-research": {
      "command": [
        "uv",
        "--directory",
        "/absolute/path/to/server",
        "run",
        "__init__.py",
        "stdio"
      ]
    },
    "chrome-devtools": {
      "command": [
        "./AID-tool-gateway/scripts/run-chrome-devtools.sh",
        "--headless=false",
        "--browserUrl=http://127.0.0.1:9222"
      ]
    },
    "linear": {
      "url": "https://mcp.linear.app/mcp"
    }
  }
}
```

For command-based servers, relative paths are resolved from `MCP/` (the folder that contains `mcporter.json`).

## Commands

```bash
cd MCP/AID-tool-gateway && npm install
cd MCP/AID-tool-gateway && npm start
cd MCP/AID-tool-gateway && npm run dev
cd MCP/AID-tool-gateway && node scripts/smoke.mjs
```
