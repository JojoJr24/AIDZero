"""Repository discovery and component cataloging."""

from __future__ import annotations

import json
from pathlib import Path

from .models import ComponentCatalog, ComponentItem


class ComponentCatalogBuilder:
    """Builds a component catalog from repository folders."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def build(self) -> ComponentCatalog:
        return ComponentCatalog(
            root=self.repo_root,
            llm_providers=self._discover_llm_providers(),
            skills=self._discover_skills(),
            tools=self._discover_tools(),
            mcp=self._discover_mcp(),
            ui=self._discover_ui(),
        )

    def _discover_llm_providers(self) -> list[ComponentItem]:
        base_dir = self.repo_root / "LLMProviders"
        items: list[ComponentItem] = []
        if not base_dir.is_dir():
            return items

        for entry in self._iter_visible_entries(base_dir):
            if not entry.is_dir():
                continue
            provider_file = entry / "provider.py"
            if not provider_file.exists():
                continue
            items.append(
                ComponentItem(
                    name=entry.name,
                    path=entry.relative_to(self.repo_root),
                    kind="llm_provider",
                    description=self._read_provider_description(provider_file),
                )
            )
        return items

    def _discover_skills(self) -> list[ComponentItem]:
        base_dir = self.repo_root / "SKILLS"
        items: list[ComponentItem] = []
        if not base_dir.is_dir():
            return items

        for entry in self._iter_visible_entries(base_dir):
            if not entry.is_dir():
                continue
            skill_doc = entry / "SKILL.md"
            if not skill_doc.exists():
                continue
            items.append(
                ComponentItem(
                    name=entry.name,
                    path=entry.relative_to(self.repo_root),
                    kind="skill",
                    description=self._read_first_meaningful_line(skill_doc),
                )
            )
        return items

    def _discover_tools(self) -> list[ComponentItem]:
        return self._discover_generic_folder(folder_name="TOOLS", kind="tool")

    def _discover_mcp(self) -> list[ComponentItem]:
        return self._discover_generic_folder(folder_name="MCP", kind="mcp")

    def _discover_ui(self) -> list[ComponentItem]:
        return self._discover_generic_folder(folder_name="UI", kind="ui")

    def _discover_generic_folder(self, *, folder_name: str, kind: str) -> list[ComponentItem]:
        base_dir = self.repo_root / folder_name
        items: list[ComponentItem] = []
        if not base_dir.is_dir():
            return items

        for entry in self._iter_visible_entries(base_dir):
            description = self._infer_folder_description(entry)
            items.append(
                ComponentItem(
                    name=entry.name,
                    path=entry.relative_to(self.repo_root),
                    kind=kind,
                    description=description,
                )
            )
        return items

    @staticmethod
    def _iter_visible_entries(base_dir: Path) -> list[Path]:
        return sorted(
            [path for path in base_dir.iterdir() if not path.name.startswith(".")],
            key=lambda path: path.name.lower(),
        )

    def _read_provider_description(self, provider_file: Path) -> str | None:
        with provider_file.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip().strip('"').strip("'")
        return first_line if first_line else None

    def _read_first_meaningful_line(self, text_file: Path) -> str | None:
        with text_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    return line.lstrip("#").strip()
                return line
        return None

    def _infer_folder_description(self, entry: Path) -> str | None:
        if entry.is_file():
            return None
        package_json = entry / "package.json"
        if package_json.exists():
            try:
                with package_json.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except json.JSONDecodeError:
                return None
            description = payload.get("description")
            if isinstance(description, str):
                return description
        readme = entry / "README.md"
        if readme.exists():
            return self._read_first_meaningful_line(readme)
        return None
