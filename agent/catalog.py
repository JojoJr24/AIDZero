"""Component catalog discovery for runtime modules."""

from __future__ import annotations

import json
from pathlib import Path

from agent.models import ComponentCatalog, ComponentItem


class ComponentCatalogBuilder:
    """Build a repository component catalog from conventional folders."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def build(self) -> ComponentCatalog:
        return ComponentCatalog(
            root=self.repo_root,
            llm_providers=self._discover_providers(),
            skills=self._discover_skills(),
            tools=self._discover_tools(),
            mcp=self._discover_mcp(),
            ui=self._discover_ui(),
        )

    def _discover_providers(self) -> list[ComponentItem]:
        providers_root = self.repo_root / "LLMProviders"
        items: list[ComponentItem] = []
        if not providers_root.is_dir():
            return items
        for entry in sorted(providers_root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir() or not entry.name.startswith("AID-"):
                continue
            provider_py = entry / "provider.py"
            if not provider_py.is_file():
                continue
            items.append(
                ComponentItem(
                    name=entry.name,
                    path=entry.relative_to(self.repo_root),
                    description=f"Provider adapter in {entry.name}",
                )
            )
        return items

    def _discover_skills(self) -> list[ComponentItem]:
        skills_root = self.repo_root / "SKILLS"
        items: list[ComponentItem] = []
        if not skills_root.is_dir():
            return items
        for entry in sorted(skills_root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            items.append(
                ComponentItem(
                    name=entry.name,
                    path=entry.relative_to(self.repo_root),
                    description=_extract_frontmatter_description(skill_md),
                )
            )
        return items

    def _discover_tools(self) -> list[ComponentItem]:
        tools_root = self.repo_root / "TOOLS"
        items: list[ComponentItem] = []
        if not tools_root.is_dir():
            return items
        for entry in sorted(tools_root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir():
                continue
            tool_py = entry / "tool.py"
            if not tool_py.is_file():
                continue
            items.append(
                ComponentItem(
                    name=entry.name,
                    path=entry.relative_to(self.repo_root),
                    description=f"Tool wrapper at {entry.name}/tool.py",
                )
            )
        return items

    def _discover_mcp(self) -> list[ComponentItem]:
        mcp_root = self.repo_root / "MCP"
        if not mcp_root.is_dir():
            return []
        items: list[ComponentItem] = []
        gateway_entry = mcp_root / "AID-tool-gateway"
        if gateway_entry.is_dir():
            items.append(
                ComponentItem(
                    name="AID-tool-gateway",
                    path=gateway_entry.relative_to(self.repo_root),
                    description="MCP stdio tool gateway",
                )
            )

        config_file = mcp_root / "mcporter.json"
        if config_file.is_file():
            server_names = _extract_mcp_server_names(config_file)
            for server_name in server_names:
                items.append(
                    ComponentItem(
                        name=server_name,
                        path=config_file.relative_to(self.repo_root),
                        description="Configured MCP server",
                    )
                )
        return items

    def _discover_ui(self) -> list[ComponentItem]:
        ui_root = self.repo_root / "UI"
        items: list[ComponentItem] = []
        if not ui_root.is_dir():
            return items
        for entry in sorted(ui_root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir():
                continue
            entrypoint_file = entry / "entrypoint.py"
            if not entrypoint_file.is_file():
                continue
            items.append(
                ComponentItem(
                    name=entry.name,
                    path=entry.relative_to(self.repo_root),
                    description=f"Runnable UI at UI/{entry.name}/entrypoint.py",
                )
            )
        return items


def _extract_frontmatter_description(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    in_frontmatter = False
    for raw_line in lines:
        line = raw_line.strip()
        if line == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter and line.startswith("description:"):
            value = line.split(":", 1)[1].strip()
            return value.strip('"').strip("'")
    return None


def _extract_mcp_server_names(config_path: Path) -> list[str]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        return []
    names = [name for name in servers.keys() if isinstance(name, str) and name.strip()]
    return sorted(names)
