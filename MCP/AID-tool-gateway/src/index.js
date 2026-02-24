import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { normalizeObjectSchema } from "@modelcontextprotocol/sdk/server/zod-compat.js";
import { toJsonSchemaCompat } from "@modelcontextprotocol/sdk/server/zod-json-schema-compat.js";
import * as z from "zod/v4";
import { createRuntime } from "mcporter";
import { ToolIndex } from "./tool-index.js";
import { SchemaValidator } from "./schema-validator.js";

const GATEWAY_NAME = "tool-search-gateway";
const GATEWAY_VERSION = "0.1.0";
const SHUTDOWN_SIGNALS = ["SIGINT", "SIGTERM"];

const toolSearchSchema = {
  query: z.string().min(1).describe("Text describing what the tool should do."),
  limit: z
    .number()
    .int()
    .min(1)
    .max(10)
    .optional()
    .describe("How many matches to return (default 5)."),
  server: z
    .string()
    .min(1)
    .optional()
    .describe("Optional substring of the server name to filter."),
  forceRefresh: z.boolean().optional().describe("Set true to bypass the cache and rescan all tools now."),
};

const toolCallSchema = {
  tool_id: z.string().min(1).describe("Identifier from tool_search results (server:tool)."),
  args: z
    .object({})
    .catchall(z.unknown())
    .optional()
    .describe("JSON arguments to forward; defaults to {}."),
  timeoutMs: z
    .number()
    .int()
    .positive()
    .optional()
    .describe("Optional timeout budget for the downstream call."),
  forceRefresh: z.boolean().optional().describe("If true, refresh catalog before resolving the tool id."),
};

const toolDescribeSchema = {
  tool_id: z.string().min(1).describe("Identifier from tool_search results (server:tool)."),
  forceRefresh: z.boolean().optional().describe("If true, refresh the catalog before describing."),
};

const toolHealthSchema = {};

async function main() {
  const runtime = await createRuntime({
    clientInfo: {
      name: `${GATEWAY_NAME}:runtime`,
      version: GATEWAY_VERSION,
    },
  });

  const toolIndex = new ToolIndex(runtime, { logger: console, excludeServers: [GATEWAY_NAME] });
  const schemaValidator = new SchemaValidator();

  const server = new McpServer(
    { name: GATEWAY_NAME, version: GATEWAY_VERSION },
    {
      instructions:
        "tool_search narrows MC tools by intent, tool_describe reveals one schema, tool_call executes via mcporter.",
    }
  );

  registerToolSearch(server, toolIndex);
  registerToolDescribe(server, toolIndex);
  registerToolCall(server, runtime, toolIndex, schemaValidator);
  registerToolHealth(server, toolIndex);

  validateRegisteredSchemas(server);

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.log(`[${GATEWAY_NAME}] ready`);

  const shutdown = async () => {
    console.log(`[${GATEWAY_NAME}] shutting down`);
    await server.close().catch((error) => console.error("server close failed", error));
    await runtime.close().catch((error) => console.error("runtime close failed", error));
    process.exit(0);
  };

  for (const signal of SHUTDOWN_SIGNALS) {
    process.on(signal, shutdown);
  }
}

function validateRegisteredSchemas(server) {
  const failures = [];
  for (const [name, tool] of Object.entries(server._registeredTools || {})) {
    try {
      const inputObj = normalizeObjectSchema(tool.inputSchema);
      if (inputObj) {
        toJsonSchemaCompat(inputObj, { strictUnions: true, pipeStrategy: "input" });
      }
      if (tool.outputSchema) {
        const outputObj = normalizeObjectSchema(tool.outputSchema);
        if (outputObj) {
          toJsonSchemaCompat(outputObj, { strictUnions: true, pipeStrategy: "output" });
        }
      }
    } catch (error) {
      failures.push({ name, error });
    }
  }

  if (failures.length) {
    const detail = failures
      .map((failure) => `${failure.name}: ${failure.error instanceof Error ? failure.error.stack ?? failure.error.message : String(failure.error)}`)
      .join("\n");
    throw new Error(`Schema serialization failed:\n${detail}`);
  }
}

function registerToolSearch(server, toolIndex) {
  server.registerTool(
    "tool_search",
    {
      title: "Tool Search",
      description: "Return up to 5 MCP tools ranked by textual relevance.",
      inputSchema: toolSearchSchema,
    },
    async (args) => {
      const { query, limit, server: serverHint, forceRefresh } = args;
      const result = await toolIndex.search(query, {
        limit,
        serverHint,
        forceRefresh,
      });

      const text = formatSearchText(query, result.matches);

      return {
        content: [
          {
            type: "text",
            text,
          },
        ],
        structuredContent: result,
      };
    }
  );
}

function registerToolCall(server, runtime, toolIndex, schemaValidator) {
  server.registerTool(
    "tool_call",
    {
      title: "Tool Call",
      description: "Execute a tool previously returned by tool_search/tool_describe.",
      inputSchema: toolCallSchema,
    },
    async (args) => {
      const forwardedArgs = ensureArgsObject(args.args);
      const timeoutMs = normalizeTimeout(args.timeoutMs);
      const toolId = typeof args.tool_id === "string" ? args.tool_id.trim() : "";
      let entry = null;
      let preparedArgs = forwardedArgs;

      try {
        entry = await toolIndex.getTool(toolId, { forceRefresh: args.forceRefresh });
        preparedArgs = schemaValidator.validate(entry.id, entry.inputSchema, forwardedArgs);

        const result = await runtime.callTool(entry.server, entry.tool, {
          args: preparedArgs,
          timeoutMs,
        });

        const forwardedContent = Array.isArray(result?.content) ? result.content : [];
        const summary = buildCallSummary({
          toolId: entry.id,
          serverName: entry.server,
          toolName: entry.tool,
          forwardedArgs: preparedArgs,
          result,
        });

        return {
          content: [
            {
              type: "text",
              text: summary,
            },
            ...forwardedContent,
          ],
          structuredContent: {
            toolId: entry.id,
            server: entry.server,
            tool: entry.tool,
            risk: entry.risk,
            args: preparedArgs,
            forwardedResult: {
              contentBlocks: forwardedContent.length,
              structuredContent: result?.structuredContent ?? null,
              isError: result?.isError ?? false,
            },
          },
          isError: result?.isError ?? false,
        };
      } catch (error) {
        const message = normalizeError(error);
        return {
          content: [
            {
              type: "text",
              text: `tool_call failed for ${toolId || "unknown"}: ${message}`,
            },
          ],
          isError: true,
          structuredContent: {
            toolId,
            server: entry?.server ?? null,
            tool: entry?.tool ?? null,
            risk: entry?.risk ?? null,
            args: preparedArgs,
            error: message,
          },
        };
      }
    }
  );
}

function registerToolDescribe(server, toolIndex) {
  server.registerTool(
    "tool_describe",
    {
      title: "Tool Describe",
      description: "Return the stored schema + metadata for a tool_id.",
      inputSchema: toolDescribeSchema,
    },
    async (args) => {
      const toolId = typeof args.tool_id === "string" ? args.tool_id.trim() : "";
      const description = await toolIndex.describe(toolId, { forceRefresh: args.forceRefresh });
      const text = formatDescribeText(description);
      return {
        content: [
          {
            type: "text",
            text,
          },
        ],
        structuredContent: description,
      };
    }
  );
}

function registerToolHealth(server, toolIndex) {
  server.registerTool(
    "tool_health",
    {
      title: "Tool Health",
      description: "Report catalog freshness + inventory stats.",
      inputSchema: toolHealthSchema,
    },
    async () => {
      const stats = toolIndex.stats();
      const text = formatHealthText(stats);
      return {
        content: [
          {
            type: "text",
            text,
          },
        ],
        structuredContent: stats,
      };
    }
  );
}

function formatSearchText(query, matches) {
  if (!matches.length) {
    return `No tools matched "${query}".`;
  }

  const header = `Top ${matches.length} matches for "${query}":`;
  const bullets = matches
    .map((match, index) => {
      const desc = match.description?.trim() || "sin descripción";
      const params = formatParameterSummary(match.parameters);
      return `${index + 1}. ${match.id} [risk:${match.risk}, score:${match.score}] ${desc}${params}`;
    })
    .join("\n");

  return `${header}\n${bullets}`;
}

function formatParameterSummary(parameters) {
  if (!parameters?.length) {
    return "";
  }
  const preview = parameters
    .map((param) => `${param.name}${param.required ? "*" : ""}`)
    .join(", ");
  return ` | params: ${preview}`;
}

function buildCallSummary({ toolId, serverName, toolName, forwardedArgs, result }) {
  const serializedArgs = Object.keys(forwardedArgs).length
    ? JSON.stringify(forwardedArgs)
    : "{}";
  const status = result?.isError ? "error" : "ok";
  return `Called ${toolId} (${serverName}/${toolName}) with ${serializedArgs} → ${status}`;
}

function formatDescribeText(description) {
  const required = description.inputSummary?.required ?? [];
  const requiredText = required.length ? required.join(", ") : "none";
  const params = description.inputSummary?.fields?.length
    ? description.inputSummary.fields.map((field) => `${field.name}:${field.type}`).join(", ")
    : "no parameters";
  return `${description.id} [risk:${description.risk}] — required: ${requiredText} | fields: ${params}`;
}

function formatHealthText(stats) {
  const refreshed = stats.refreshedAt ?? "never";
  return `Indexed tools: ${stats.totalIndexedTools} | last refresh: ${refreshed} | ttl: ${stats.refreshIntervalMs}ms`;
}

function ensureArgsObject(args) {
  if (args === undefined || args === null) {
    return {};
  }

  if (typeof args !== "object" || Array.isArray(args)) {
    throw new Error("tool_call args must be an object");
  }

  return args;
}

function normalizeTimeout(value) {
  if (value === undefined || value === null) {
    return undefined;
  }

  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    throw new Error("timeoutMs must be a positive number when provided");
  }

  return numeric;
}

function normalizeError(error) {
  if (!error) {
    return "unknown error";
  }

  if (error instanceof Error) {
    return error.message;
  }

  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}

main().catch((error) => {
  console.error(`[${GATEWAY_NAME}] fatal`, error);
  process.exit(1);
});
