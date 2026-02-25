#!/usr/bin/env node
// Purpose: smoke-test tool gateway via STDIO transport.
// Usage: node scripts/smoke.mjs
// Outputs: logs basic listTools + tool_health responses.

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..", "..");
const gatewayLauncher = path.join(repoRoot, "MCP", "run-tool-gateway.sh");

async function main() {
  const transport = new StdioClientTransport({
    command: gatewayLauncher,
  });

  const client = new Client({ name: "tool-gateway-smoke", version: "0.0.1" });

  try {
    console.log("connecting to gateway...");
    await client.connect(transport);

    console.log("requesting tools/list...");
    const listResult = await client.listTools();
    console.log("tools/list", listResult.tools.map((tool) => tool.name));

    console.log("calling tool_health...");
    const healthResult = await client.callTool({ name: "tool_health", arguments: {} });
    console.log("tool_health", healthResult.structuredContent ?? healthResult);
  } finally {
    await client.close().catch(() => {});
    await transport.close().catch(() => {});
  }
}

main().catch((error) => {
  console.error("Smoke test failed", error);
  process.exit(1);
});
