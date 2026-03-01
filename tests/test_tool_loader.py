from __future__ import annotations

from agent.memory import MemoryStore
from agent.tooling import build_default_tool_registry


def test_build_default_tool_registry_loads_tools_from_tools_folder(tmp_path):
    tools_dir = tmp_path / "TOOLS"
    tools_dir.mkdir(parents=True, exist_ok=True)

    (tools_dir / "echo_tool.py").write_text(
        "\n".join(
            [
                'TOOL_NAME = "echo"',
                'TOOL_DESCRIPTION = "Echo input."',
                'TOOL_PARAMETERS = {"type": "object", "properties": {"value": {"type": "string"}}}',
                "",
                "def run(arguments, *, repo_root, memory):",
                "    del repo_root, memory",
                "    return {'echo': arguments.get('value')}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    memory = MemoryStore(tmp_path / ".aidzero" / "memory.json")
    registry = build_default_tool_registry(tmp_path, memory)

    assert "echo" in registry.names()
    assert registry.execute("echo", {"value": "ok"}) == {"echo": "ok"}
