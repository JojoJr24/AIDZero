#!/usr/bin/env node
// Purpose: call one gateway tool over stdio and print structured JSON output.
// Usage: node scripts/gateway-call.mjs --tool tool_search --payload '{"query":"..."}'

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..", "..");
const gatewayLauncher = path.join(repoRoot, "MCP", "run-tool-gateway.sh");
const SHUTDOWN_TIMEOUT_MS = 2000;

function parseArgs(argv) {
  let toolName = "";
  let payloadRaw = "{}";
  for (let index = 2; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--tool" && index + 1 < argv.length) {
      toolName = argv[index + 1];
      index += 1;
      continue;
    }
    if (token === "--payload" && index + 1 < argv.length) {
      payloadRaw = argv[index + 1];
      index += 1;
      continue;
    }
  }

  if (!toolName.trim()) {
    throw new Error("Missing required --tool argument.");
  }

  let payload = {};
  try {
    payload = JSON.parse(payloadRaw);
  } catch (error) {
    throw new Error(`Invalid JSON passed to --payload: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("--payload must be a JSON object.");
  }
  return { toolName: toolName.trim(), payload };
}

async function main() {
  const { toolName, payload } = parseArgs(process.argv);
  const transport = new StdioClientTransport({ command: gatewayLauncher });
  const client = new Client({ name: "gateway-call", version: "0.0.1" });

  try {
    await client.connect(transport);
    const result = await client.callTool({ name: toolName, arguments: payload });
    process.stdout.write(
      `${JSON.stringify(
        {
          tool: toolName,
          isError: Boolean(result?.isError),
          structuredContent: result?.structuredContent ?? null,
          content: Array.isArray(result?.content) ? result.content : [],
        },
        null,
        2
      )}\n`
    );
  } finally {
    await closeWithTimeout(() => client.close(), SHUTDOWN_TIMEOUT_MS);
    await closeWithTimeout(() => transport.close(), SHUTDOWN_TIMEOUT_MS);
  }
}

async function closeWithTimeout(closeFn, timeoutMs) {
  await Promise.race([
    Promise.resolve()
      .then(() => closeFn())
      .catch(() => {}),
    new Promise((resolve) => {
      const timer = setTimeout(resolve, timeoutMs);
      if (typeof timer.unref === "function") {
        timer.unref();
      }
    }),
  ]);
}

main()
  .then(() => {
    process.exit(0);
  })
  .catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exit(1);
  });
