import { describeConnectionIssue } from "mcporter";

const DEFAULT_REFRESH_INTERVAL_MS = 30_000;
const DEFAULT_RESULT_LIMIT = 5;

/**
 * Maintains an in-memory index of all known MCP tools using mcporter as a backend.
 */
export class ToolIndex {
  constructor(runtime, options = {}) {
    this.runtime = runtime;
    this.refreshIntervalMs = options.refreshIntervalMs ?? DEFAULT_REFRESH_INTERVAL_MS;
    this.logger = options.logger ?? console;
    this.entries = [];
    this.entryMap = new Map();
    this.excludedServers = new Set(options.excludeServers ?? []);
    this.lastRefresh = 0;
    this.refreshPromise = null;
  }

  async ensureFresh(force = false) {
    const needsRefresh =
      force ||
      !this.entries.length ||
      Date.now() - this.lastRefresh > this.refreshIntervalMs;

    if (!needsRefresh) {
      return;
    }

    if (!this.refreshPromise) {
      this.refreshPromise = this.pullLatest()
        .catch((error) => {
          this.logger.error("[tool-index] refresh failed", error);
          throw error;
        })
        .finally(() => {
          this.refreshPromise = null;
        });
    }

    return this.refreshPromise;
  }

  async pullLatest() {
    const servers = this.runtime.listServers();
    const snapshot = [];

    for (const serverName of servers) {
      if (this.excludedServers.has(serverName)) {
        continue;
      }
      try {
        const tools = await this.runtime.listTools(serverName, {
          includeSchema: true,
          autoAuthorize: true,
        });

        for (const tool of tools) {
          snapshot.push(buildEntry(serverName, tool));
        }
      } catch (error) {
        const issue = describeConnectionIssue(error);
        this.logger.warn(
          `[tool-index] Unable to list tools for ${serverName}: ${issue.message ?? error.message}`
        );
      }
    }

    this.entries = snapshot;
    this.entryMap = new Map(snapshot.map((entry) => [entry.id, entry]));
    this.lastRefresh = Date.now();
    return snapshot;
  }

  async search(query, options = {}) {
    const normalizedQuery = normalizeQuery(query);
    if (!normalizedQuery) {
      throw new Error("Query text is required");
    }

    await this.ensureFresh(options.forceRefresh ?? false);

    const tokens = normalizedQuery.split(/\s+/).filter(Boolean);
    const limit = clampCount(options.limit ?? DEFAULT_RESULT_LIMIT);
    const serverHint = options.serverHint?.toLowerCase?.() ?? null;

    const scored = this.entries
      .filter((entry) =>
        serverHint ? entry.serverNormalized.includes(serverHint) : true
      )
      .map((entry) => ({
        entry,
        score: computeScore(entry, tokens, normalizedQuery),
      }))
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map(({ entry, score }) => formatMatch(entry, score));

    return {
      matches: scored,
      refreshedAt: new Date(this.lastRefresh).toISOString(),
      totalIndexedTools: this.entries.length,
    };
  }

  async describe(toolId, options = {}) {
    const entry = await this.getTool(toolId, options);
    return formatDescription(entry);
  }

  async getTool(toolId, options = {}) {
    const normalizedId = normalizeToolId(toolId);
    if (!normalizedId) {
      throw new Error("tool_id is required");
    }

    await this.ensureFresh(options.forceRefresh ?? false);
    let entry = this.entryMap.get(normalizedId);

    if (!entry && options.forceRefresh !== true) {
      await this.ensureFresh(true);
      entry = this.entryMap.get(normalizedId);
    }

    if (!entry) {
      throw new Error(`Unknown tool_id "${toolId}"`);
    }

    return entry;
  }

  stats() {
    return {
      totalIndexedTools: this.entries.length,
      refreshedAt: this.lastRefresh ? new Date(this.lastRefresh).toISOString() : null,
      refreshIntervalMs: this.refreshIntervalMs,
    };
  }
}

function buildEntry(serverName, toolInfo) {
  const normalizedDescription = (toolInfo.description ?? "").toLowerCase();
  const normalizedServer = serverName.toLowerCase();
  const normalizedName = toolInfo.name.toLowerCase();
  const inputSummary = summarizeSchema(toolInfo.inputSchema);
  const outputSummary = summarizeSchema(toolInfo.outputSchema);
  const schemaTokens = buildSchemaTokens(inputSummary, outputSummary);
  const risk = classifyRisk(toolInfo.name, toolInfo.description);

  return {
    id: `${serverName}:${toolInfo.name}`,
    server: serverName,
    tool: toolInfo.name,
    description: toolInfo.description ?? null,
    inputSchema: toolInfo.inputSchema ?? null,
    outputSchema: toolInfo.outputSchema ?? null,
    serverNormalized: normalizedServer,
    toolNormalized: normalizedName,
    descriptionNormalized: normalizedDescription,
    schemaTokens,
    inputSummary,
    outputSummary,
    risk,
  };
}

function summarizeSchema(schema) {
  if (!schema || typeof schema !== "object") {
    return null;
  }

  const required = new Set(Array.isArray(schema.required) ? schema.required : []);
  const properties = schema.properties && typeof schema.properties === "object" ? schema.properties : {};
  const fields = Object.entries(properties).map(([name, value]) => ({
    name,
    type: inferSchemaType(value),
    required: required.has(name),
    description:
      value && typeof value === "object" && typeof value.description === "string"
        ? value.description
        : undefined,
  }));

  return {
    type: typeof schema.type === "string" ? schema.type : "object",
    fields,
    required: [...required],
  };
}

function inferSchemaType(node) {
  if (!node || typeof node !== "object") {
    return "unknown";
  }

  if (typeof node.type === "string") {
    return node.type;
  }

  if (Array.isArray(node.type)) {
    return node.type.join("|");
  }

  if (Array.isArray(node.enum)) {
    return `enum(${node.enum.length})`;
  }

  if (node.properties) {
    return "object";
  }

  if (node.items) {
    return "array";
  }

  return "unknown";
}

function buildSchemaTokens(...summaries) {
  return summaries
    .filter(Boolean)
    .flatMap((summary) => summary.fields.map((field) => `${field.name.toLowerCase()} ${field.type}`));
}

const DESTRUCTIVE_KEYWORDS = [
  "delete",
  "destroy",
  "terminate",
  "drop",
  "wipe",
  "revoke",
  "shutdown",
  "disable",
  "remove",
  "kill",
];

const WRITE_KEYWORDS = [
  "create",
  "update",
  "write",
  "send",
  "post",
  "patch",
  "put",
  "start",
  "open",
  "modify",
];

function classifyRisk(name, description) {
  const haystack = `${name ?? ""} ${description ?? ""}`.toLowerCase();
  if (matchesKeyword(haystack, DESTRUCTIVE_KEYWORDS)) {
    return "destructive";
  }
  if (matchesKeyword(haystack, WRITE_KEYWORDS)) {
    return "write";
  }
  return "read";
}

function matchesKeyword(text, keywords) {
  return keywords.some((keyword) => text.includes(keyword));
}

function computeScore(entry, tokens, fallbackQuery) {
  if (!tokens.length) {
    return baseScore(entry, fallbackQuery);
  }

  return tokens.reduce((acc, token) => {
    return (
      acc +
      weightMatch(entry.toolNormalized, token, 4) +
      weightMatch(entry.descriptionNormalized, token, 2) +
      weightMatch(entry.serverNormalized, token, 1) +
      weightSchema(entry.schemaTokens, token)
    );
  }, 0);
}

function baseScore(entry, fallbackQuery) {
  const summaryBlob = [entry.toolNormalized, entry.descriptionNormalized, entry.serverNormalized].join(" ");
  return similarity(summaryBlob, fallbackQuery) * 2;
}

function weightMatch(haystack, needle, multiplier) {
  if (!haystack || !needle) {
    return 0;
  }

  const idx = haystack.indexOf(needle);
  if (idx === -1) {
    return 0;
  }

  const lengthBonus = needle.length / (haystack.length + 1);
  const positionBonus = 1 - idx / (haystack.length + 1);
  return multiplier * (1 + lengthBonus + positionBonus);
}

function weightSchema(tokens, needle) {
  if (!tokens?.length || !needle) {
    return 0;
  }

  let score = 0;
  for (const token of tokens) {
    if (token.includes(needle)) {
      score += 0.5 + needle.length / (token.length + 1);
    }
  }
  return score;
}

function similarity(a, b) {
  if (!a || !b) {
    return 0;
  }

  const shorter = a.length < b.length ? a : b;
  const longer = a.length >= b.length ? a : b;
  let matches = 0;

  for (const char of shorter) {
    if (longer.includes(char)) {
      matches += 1;
    }
  }

  return matches / longer.length;
}

function normalizeQuery(value) {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

function clampCount(value) {
  if (!Number.isFinite(value)) {
    return DEFAULT_RESULT_LIMIT;
  }

  return Math.max(1, Math.min(10, Math.floor(value)));
}

function formatMatch(entry, score) {
  return {
    id: entry.id,
    server: entry.server,
    tool: entry.tool,
    description: entry.description,
    score: Number(score.toFixed(3)),
    risk: entry.risk,
    parameters: summarizeParameters(entry.inputSummary),
  };
}

function formatDescription(entry) {
  return {
    id: entry.id,
    server: entry.server,
    tool: entry.tool,
    description: entry.description,
    inputSchema: entry.inputSchema,
    outputSchema: entry.outputSchema,
    inputSummary: entry.inputSummary,
    outputSummary: entry.outputSummary,
    risk: entry.risk,
  };
}

function summarizeParameters(summary) {
  if (!summary?.fields?.length) {
    return [];
  }

  return summary.fields.slice(0, 5).map((field) => ({
    name: field.name,
    type: field.type,
    required: field.required,
  }));
}

function normalizeToolId(value) {
  if (typeof value !== "string") {
    return "";
  }
  return value.trim();
}
