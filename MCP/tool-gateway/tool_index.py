"""In-memory MCP tool index with lightweight ranking and risk classification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_REFRESH_INTERVAL_MS = 30_000
DEFAULT_RESULT_LIMIT = 5
DISCOVERY_QUERIES = {
    "*",
    "all",
    "all tools",
    "available tools",
    "discover tools",
    "list",
    "list available tools",
    "list tools",
    "show tools",
    "tools",
}

DESTRUCTIVE_KEYWORDS = {
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
}

WRITE_KEYWORDS = {
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
}


@dataclass(slots=True)
class ToolEntry:
    id: str
    server: str
    tool: str
    description: str | None
    input_schema: dict[str, Any] | None
    output_schema: dict[str, Any] | None
    server_normalized: str
    tool_normalized: str
    description_normalized: str
    schema_tokens: list[str]
    input_summary: dict[str, Any] | None
    output_summary: dict[str, Any] | None
    risk: str


class ToolIndex:
    def __init__(
        self,
        runtime,
        *,
        refresh_interval_ms: int = DEFAULT_REFRESH_INTERVAL_MS,
        exclude_servers: list[str] | None = None,
        server_list_timeout_seconds: float = 8.0,
    ) -> None:
        self.runtime = runtime
        self.refresh_interval_ms = max(1, int(refresh_interval_ms))
        self._exclude_servers = set(exclude_servers or [])
        self.server_list_timeout_seconds = max(0.5, float(server_list_timeout_seconds))
        self.entries: list[ToolEntry] = []
        self.entry_map: dict[str, ToolEntry] = {}
        self.last_refresh_ms: int = 0

    async def ensure_fresh(self, force: bool = False) -> None:
        now_ms = _now_ms()
        needs_refresh = force or not self.entries or (now_ms - self.last_refresh_ms > self.refresh_interval_ms)
        if needs_refresh:
            await self.pull_latest()

    async def pull_latest(self) -> list[ToolEntry]:
        snapshot: list[ToolEntry] = []
        for server_name in self.runtime.list_servers():
            if server_name in self._exclude_servers:
                continue
            try:
                tools = await self.runtime.list_tools(
                    server_name,
                    include_schema=True,
                    timeout_seconds=self.server_list_timeout_seconds,
                )
            except Exception:
                continue
            for tool in tools:
                snapshot.append(_build_entry(server_name, tool))

        self.entries = snapshot
        self.entry_map = {entry.id: entry for entry in snapshot}
        self.last_refresh_ms = _now_ms()
        return snapshot

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        server_hint: str | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        normalized_query = _normalize_query(query)
        if not normalized_query:
            raise ValueError("Query text is required")

        await self.ensure_fresh(force=force_refresh)

        tokens = [token for token in normalized_query.split() if token]
        discovery_mode = _is_discovery_query(normalized_query)
        capped_limit = _clamp_count(limit if limit is not None else DEFAULT_RESULT_LIMIT)
        normalized_server_hint = server_hint.lower() if isinstance(server_hint, str) else None

        scored: list[tuple[ToolEntry, float]] = []
        for entry in self.entries:
            if normalized_server_hint and normalized_server_hint not in entry.server_normalized:
                continue
            if discovery_mode:
                scored.append((entry, 1.0))
                continue
            score = _compute_score(entry, tokens, normalized_query)
            if score > 0:
                scored.append((entry, score))

        if discovery_mode:
            scored.sort(key=lambda item: (item[0].server_normalized, item[0].tool_normalized))
        else:
            scored.sort(key=lambda item: item[1], reverse=True)
        matches = [_format_match(entry, score) for entry, score in scored[:capped_limit]]

        return {
            "matches": matches,
            "refreshedAt": _iso_from_ms(self.last_refresh_ms),
            "totalIndexedTools": len(self.entries),
        }

    async def describe(self, tool_id: str, *, force_refresh: bool = False) -> dict[str, Any]:
        entry = await self.get_tool(tool_id, force_refresh=force_refresh)
        return _format_description(entry)

    async def get_tool(self, tool_id: str, *, force_refresh: bool = False) -> ToolEntry:
        normalized = _normalize_tool_id(tool_id)
        if not normalized:
            raise ValueError("tool_id is required")

        await self.ensure_fresh(force=force_refresh)
        entry = self.entry_map.get(normalized)

        if entry is None and not force_refresh:
            await self.ensure_fresh(force=True)
            entry = self.entry_map.get(normalized)

        if entry is None:
            raise ValueError(f"Unknown tool_id '{tool_id}'")

        return entry

    def stats(self) -> dict[str, Any]:
        return {
            "totalIndexedTools": len(self.entries),
            "refreshedAt": _iso_from_ms(self.last_refresh_ms) if self.last_refresh_ms else None,
            "refreshIntervalMs": self.refresh_interval_ms,
        }


def _build_entry(server_name: str, tool_info: dict[str, Any]) -> ToolEntry:
    description = tool_info.get("description") if isinstance(tool_info.get("description"), str) else None
    tool_name = str(tool_info.get("name", "")).strip()
    input_schema = tool_info.get("inputSchema") if isinstance(tool_info.get("inputSchema"), dict) else None
    output_schema = tool_info.get("outputSchema") if isinstance(tool_info.get("outputSchema"), dict) else None

    input_summary = _summarize_schema(input_schema)
    output_summary = _summarize_schema(output_schema)

    return ToolEntry(
        id=f"{server_name}:{tool_name}",
        server=server_name,
        tool=tool_name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        server_normalized=server_name.lower(),
        tool_normalized=tool_name.lower(),
        description_normalized=(description or "").lower(),
        schema_tokens=_build_schema_tokens(input_summary, output_summary),
        input_summary=input_summary,
        output_summary=output_summary,
        risk=_classify_risk(tool_name, description or ""),
    )


def _summarize_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(schema, dict):
        return None

    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    required_set = {str(item) for item in required if isinstance(item, str)}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}

    fields: list[dict[str, Any]] = []
    for name, value in properties.items():
        if not isinstance(name, str):
            continue
        node = value if isinstance(value, dict) else {}
        description = node.get("description") if isinstance(node.get("description"), str) else None
        fields.append(
            {
                "name": name,
                "type": _infer_schema_type(node),
                "required": name in required_set,
                "description": description,
            }
        )

    return {
        "type": schema.get("type") if isinstance(schema.get("type"), str) else "object",
        "fields": fields,
        "required": sorted(required_set),
    }


def _infer_schema_type(node: dict[str, Any]) -> str:
    node_type = node.get("type")
    if isinstance(node_type, str):
        return node_type
    if isinstance(node_type, list):
        values = [str(item) for item in node_type]
        return "|".join(values)
    if isinstance(node.get("enum"), list):
        return f"enum({len(node['enum'])})"
    if isinstance(node.get("properties"), dict):
        return "object"
    if "items" in node:
        return "array"
    return "unknown"


def _build_schema_tokens(*summaries: dict[str, Any] | None) -> list[str]:
    tokens: list[str] = []
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        fields = summary.get("fields") if isinstance(summary.get("fields"), list) else []
        for field in fields:
            if not isinstance(field, dict):
                continue
            name = str(field.get("name", "")).lower().strip()
            field_type = str(field.get("type", "")).strip()
            if name:
                tokens.append(f"{name} {field_type}")
    return tokens


def _classify_risk(name: str, description: str) -> str:
    haystack = f"{name} {description}".lower()
    if any(keyword in haystack for keyword in DESTRUCTIVE_KEYWORDS):
        return "destructive"
    if any(keyword in haystack for keyword in WRITE_KEYWORDS):
        return "write"
    return "read"


def _compute_score(entry: ToolEntry, tokens: list[str], fallback_query: str) -> float:
    if not tokens:
        return _base_score(entry, fallback_query)

    score = 0.0
    for token in tokens:
        score += _weight_match(entry.tool_normalized, token, 4.0)
        score += _weight_match(entry.description_normalized, token, 2.0)
        score += _weight_match(entry.server_normalized, token, 1.0)
        score += _weight_schema(entry.schema_tokens, token)
    return score


def _base_score(entry: ToolEntry, fallback_query: str) -> float:
    blob = " ".join([entry.tool_normalized, entry.description_normalized, entry.server_normalized])
    return _similarity(blob, fallback_query) * 2.0


def _weight_match(haystack: str, needle: str, multiplier: float) -> float:
    if not haystack or not needle:
        return 0.0
    idx = haystack.find(needle)
    if idx == -1:
        return 0.0
    length_bonus = len(needle) / (len(haystack) + 1)
    position_bonus = 1 - idx / (len(haystack) + 1)
    return multiplier * (1 + length_bonus + position_bonus)


def _weight_schema(tokens: list[str], needle: str) -> float:
    if not tokens or not needle:
        return 0.0
    score = 0.0
    for token in tokens:
        if needle in token:
            score += 0.5 + len(needle) / (len(token) + 1)
    return score


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    matches = sum(1 for char in shorter if char in longer)
    return matches / max(1, len(longer))


def _normalize_query(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _is_discovery_query(query: str) -> bool:
    normalized = " ".join(query.split())
    return normalized in DISCOVERY_QUERIES


def _clamp_count(value: Any) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return DEFAULT_RESULT_LIMIT
    return max(1, min(10, numeric))


def _format_match(entry: ToolEntry, score: float) -> dict[str, Any]:
    return {
        "id": entry.id,
        "server": entry.server,
        "tool": entry.tool,
        "description": entry.description,
        "score": round(score, 3),
        "risk": entry.risk,
        "parameters": _summarize_parameters(entry.input_summary),
    }


def _format_description(entry: ToolEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "server": entry.server,
        "tool": entry.tool,
        "description": entry.description,
        "inputSchema": entry.input_schema,
        "outputSchema": entry.output_schema,
        "inputSummary": entry.input_summary,
        "outputSummary": entry.output_summary,
        "risk": entry.risk,
    }


def _summarize_parameters(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(summary, dict):
        return []
    fields = summary.get("fields") if isinstance(summary.get("fields"), list) else []
    out: list[dict[str, Any]] = []
    for field in fields[:5]:
        if not isinstance(field, dict):
            continue
        out.append(
            {
                "name": str(field.get("name", "")),
                "type": str(field.get("type", "unknown")),
                "required": bool(field.get("required", False)),
            }
        )
    return out


def _normalize_tool_id(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _iso_from_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
