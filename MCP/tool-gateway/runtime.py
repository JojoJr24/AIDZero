"""Runtime helpers to talk to MCP servers declared in MCP/mcporter.json."""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, AsyncIterator

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Implementation


@dataclass(slots=True)
class ServerConfig:
    name: str
    raw: dict[str, Any]


class McpRuntime:
    """Thin runtime to list and call tools across configured MCP servers."""

    def __init__(self, *, config_path: Path, client_name: str, client_version: str) -> None:
        self.config_path = config_path
        self.config_dir = config_path.parent
        self.client_info = Implementation(name=client_name, version=client_version)
        self._servers = self._load_server_configs()

    def list_servers(self) -> list[str]:
        return [config.name for config in self._servers]

    async def list_tools(
        self,
        server_name: str,
        *,
        include_schema: bool = True,
        timeout_seconds: float | None = None,
    ) -> list[dict[str, Any]]:
        del include_schema
        timeout = _normalize_timeout(timeout_seconds)
        if timeout is None:
            async with self.session_for(server_name) as session:
                response = await session.list_tools()
        else:
            with anyio.fail_after(timeout):
                async with self.session_for(server_name) as session:
                    response = await session.list_tools()
        tools: list[dict[str, Any]] = []
        for tool in response.tools:
            tools.append(tool.model_dump(exclude_none=True))
        return tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        *,
        args: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        timeout_seconds = _timeout_seconds_from_ms(timeout_ms)
        if timeout_seconds is None:
            async with self.session_for(server_name) as session:
                result = await session.call_tool(tool_name, args or {})
        else:
            with anyio.fail_after(timeout_seconds):
                async with self.session_for(server_name) as session:
                    result = await session.call_tool(tool_name, args or {})
        return result.model_dump(exclude_none=True)

    @asynccontextmanager
    async def session_for(self, server_name: str) -> AsyncIterator[ClientSession]:
        config = self._get_server(server_name)
        async with AsyncExitStack() as stack:
            read_stream = None
            write_stream = None

            if isinstance(config.raw.get("url"), str) and config.raw.get("url", "").strip():
                url = str(config.raw["url"]).strip()
                read_stream, write_stream, _ = await stack.enter_async_context(streamablehttp_client(url))
            else:
                command = config.raw.get("command")
                params = self._build_stdio_params(command, config.raw.get("env"))
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))

            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    client_info=self.client_info,
                )
            )
            await session.initialize()
            yield session

    def _build_stdio_params(self, command: Any, env_overrides: Any) -> StdioServerParameters:
        if isinstance(command, str):
            tokens = [command]
        elif isinstance(command, list) and all(isinstance(token, str) for token in command):
            tokens = [token for token in command if token]
        else:
            raise ValueError("MCP server command must be a string or list of strings")

        if not tokens:
            raise ValueError("MCP server command is empty")

        resolved_command = self._resolve_command_token(tokens[0])
        resolved_args = [self._resolve_command_token(token) for token in tokens[1:]]

        env: dict[str, str] | None = None
        if isinstance(env_overrides, dict):
            env = {str(key): str(value) for key, value in env_overrides.items()}

        return StdioServerParameters(
            command=resolved_command,
            args=resolved_args,
            env=env,
            cwd=str(self.config_dir),
        )

    def _resolve_command_token(self, token: str) -> str:
        if token.startswith("./") or token.startswith("../"):
            return str((self.config_dir / token).resolve())
        if "/" in token and not token.startswith("/"):
            return str((self.config_dir / token).resolve())
        return token

    def _load_server_configs(self) -> list[ServerConfig]:
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return []
        servers = payload.get("mcpServers")
        if not isinstance(servers, dict):
            return []

        configs: list[ServerConfig] = []
        for name, raw in servers.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(raw, dict):
                continue
            enabled = raw.get("enabled", True)
            if enabled is False:
                continue
            configs.append(ServerConfig(name=name.strip(), raw=raw))
        return configs

    def _get_server(self, server_name: str) -> ServerConfig:
        normalized = server_name.strip()
        for config in self._servers:
            if config.name == normalized:
                return config
        raise ValueError(f"Unknown server '{server_name}'.")


def _normalize_timeout(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    try:
        value = float(timeout_seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _timeout_seconds_from_ms(timeout_ms: int | None) -> float | None:
    if timeout_ms is None:
        return None
    try:
        numeric = int(timeout_ms)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric / 1000.0
