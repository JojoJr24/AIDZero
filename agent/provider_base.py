"""Provider-base runtime that wires prompt, tools, skills, and MCP gateway."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable

from LLMProviders.provider_base import LLMProvider, normalize_tool_result_content, parse_json_object
from agent.models import ComponentCatalog, ComponentItem

ToolExecutor = Callable[[dict[str, Any]], Any]
InvocationTracer = Callable[[dict[str, Any]], None]
MCP_GATEWAY_CALL_TIMEOUT_SECONDS = 45


@dataclass(frozen=True)
class ToolSpec:
    """Tool definition + local executor."""

    definition: dict[str, Any]
    execute: ToolExecutor


class ProviderBaseRuntime:
    """Run one chat exchange through a provider with tool/mcp/skill context."""

    def __init__(self, *, provider: LLMProvider, model: str, repo_root: Path, catalog: ComponentCatalog) -> None:
        self.provider = provider
        self.model = model.strip()
        self.repo_root = repo_root.resolve()
        self.catalog = catalog
        self._tool_specs = self._build_tool_specs()

    def ask(
        self,
        *,
        prompt: str,
        ui_name: str | None = None,
        max_tool_rounds: int = 4,
        invocation_tracer: InvocationTracer | None = None,
    ) -> str:
        user_prompt = prompt.strip()
        if not user_prompt:
            raise ValueError("prompt cannot be empty.")
        system_prompt = self._build_system_prompt(ui_name=ui_name)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        assembled_chunks: list[str] = []
        rounds = 0
        awaiting_tool_continuation = False
        continuation_attempts = 0
        max_continuation_attempts = 8
        while True:
            request_messages = [dict(message) for message in messages]
            stream_result = self._stream_until_tool_call(messages)
            if invocation_tracer is not None:
                invocation_payload: dict[str, Any] = {
                    "round": rounds + 1,
                    "model": self.model,
                    "messages": request_messages,
                    "response_text": stream_result.text,
                    "tool_call": stream_result.tool_call,
                    "finished": stream_result.tool_call is None,
                }
                invocation_tracer(invocation_payload)
            if stream_result.text:
                assembled_chunks.append(stream_result.text)
            if stream_result.tool_call is None:
                has_text = bool(stream_result.text.strip())
                if awaiting_tool_continuation and not has_text:
                    fallback_text = self._complete_with_non_streaming(messages)
                    if fallback_text:
                        assembled_chunks.append(fallback_text)
                        awaiting_tool_continuation = False
                        continuation_attempts = 0
                        return "".join(assembled_chunks).strip()
                    continuation_attempts += 1
                    if continuation_attempts >= max_continuation_attempts:
                        raise RuntimeError(
                            "Model did not provide a final response after tool output."
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Tool output was already provided (including errors if any). "
                                "Continue and finish your response now."
                            ),
                        }
                    )
                    continue
                awaiting_tool_continuation = False
                continuation_attempts = 0
                return "".join(assembled_chunks).strip()

            rounds += 1
            if rounds > max_tool_rounds:
                return ("".join(assembled_chunks) + "\nTool call limit reached.").strip()

            call_name = stream_result.tool_call["name"]
            call_arguments = stream_result.tool_call["arguments"]
            call_block = stream_result.tool_call["raw_block"]
            assistant_content = "\n".join(
                chunk for chunk in (stream_result.text.strip(), call_block.strip()) if chunk
            ).strip()
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            tool_result = self._execute_tool(call_name, call_arguments)
            tool_failed = isinstance(tool_result, dict) and "error" in tool_result
            tool_context = {
                "tool_name": call_name,
                "status": "error" if tool_failed else "ok",
                "result": tool_result,
            }
            tool_result_content = normalize_tool_result_content(tool_context)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Response:\n"
                        f"{tool_result_content}\n\n"
                        "Continue your previous answer using this response."
                    ),
                }
            )
            awaiting_tool_continuation = True
            continuation_attempts = 0

    def _complete_with_non_streaming(self, messages: list[dict[str, Any]]) -> str:
        try:
            payload = self.provider.chat(self.model, messages)
        except Exception:  # noqa: BLE001
            return ""
        return _extract_response_text(payload).strip()

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        spec = self._tool_specs.get(name)
        if spec is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return spec.execute(arguments)
        except Exception as error:  # noqa: BLE001
            return {"error": str(error)}

    def _build_system_prompt(self, *, ui_name: str | None) -> str:
        active_ui = (ui_name or "unknown").strip() or "unknown"
        provider_inventory = self._format_component_inventory(
            self.catalog.llm_providers,
            default_description="Provider adapter available to the runtime.",
        )
        local_tool_inventory = self._format_local_tool_inventory()
        mcp_inventory = self._format_component_inventory(
            self.catalog.mcp,
            default_description="MCP component available through the gateway/runtime.",
        )
        skill_tool_runtime_name = _safe_tool_name("aid_tool_AID-skill-tool")
        tool_schema = json.dumps(self._tool_prompt_payload(), ensure_ascii=False, indent=2)

        skills: list[str] = []
        for item in self.catalog.skills:
            skill_md = self.repo_root / item.path / "SKILL.md"
            body = _read_text(skill_md, fallback="")
            snippet = body[:1200].strip()
            if snippet:
                skills.append(f"[{item.name}]\n{snippet}")
            else:
                skills.append(f"[{item.name}]")
        skills_blob = "\n\n".join(skills) if skills else "No skills loaded."
        return (
            "You are AIDZero, the parent agent creator. "
            "You must write child agents.\n"
            "Your deliverable is implementation-oriented: generate the Python code design/content for the new child "
            "agent so it can run, invoke LLMs, and solve its target task.\n"
            "Child agent baseline: start from a scaffold copy of the parent runtime/repository, "
            "then keep only the providers/tools/MCP components required for the user request.\n"
            "Critical scope rule: all implementation instructions apply to the newly scaffolded child workspace, "
            "never to the parent repository.\n"
            "Child entrypoint rule: keep a Python entry file at the child root (`main.py`) and wire it to run the "
            "child runtime.\n"
            "Child runtime rule: put child-agent logic inside the child `agent/` package (for example `agent/*.py`); "
            "do not place child logic in parent files.\n"
            "The child must use the available parent scaffolding (providers, local tools, MCP gateway, skills, UI "
            "runtime patterns) as reusable building blocks instead of reinventing them.\n"
            "Tool policy: parent can use all available tools to reason and discover information; "
            "child must keep only the subset it needs.\n"
            "Selection workflow is mandatory before you define child requirements:\n"
            "1) Inspect parent capabilities from provider/tool/MCP inventories and their descriptions.\n"
            "2) Inspect available skills (loaded snapshot first, then skill tools if needed).\n"
            "3) Decide child components with include/exclude reasoning; never assume the child needs everything.\n"
            "4) Keep in the child only components that directly support the user goal.\n"
            "Skills are knowledge guides: each skill provides task guidance and descriptions that help you obtain "
            "the missing domain/implementation knowledge before choosing child components.\n"
            "If you do not have enough context, discover skills before guessing. "
            f"Prefer the local tool AID-skill-tool (call name: {skill_tool_runtime_name}) "
            "or use aid_list_skills + aid_read_skill.\n"
            "Answer the user prompt from the active UI and use tools when needed.\n"
            f"Active UI: {active_ui}\n"
            f"Registered providers (name: description):\n{provider_inventory}\n"
            f"Registered local tools (name/call: description):\n{local_tool_inventory}\n"
            f"Registered MCP components (name: description):\n{mcp_inventory}\n"
            f"Loaded skills snapshot:\n{skills_blob}\n"
            "Tools are available only through this JSON schema:\n"
            f"{tool_schema}\n"
            "To invoke a tool, emit exactly one block with this unique format:\n"
            "<AID_TOOL_CALL>\n"
            '{"name":"aid_list_skills","arguments":{}}\n'
            "</AID_TOOL_CALL>\n"
            "Do not use markdown code fences for tool calls."
            " After emitting that block, stop generating text and wait for a follow-up `Response:` "
            "message that contains the tool output. Then continue your answer."
            "\nNever call tools through provider-native function calling."
            "\nDo not wrap normal prose inside `<AID_TOOL_CALL>` blocks."
            "\nPrefer concise and actionable responses."
        )

    def _tool_prompt_payload(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for spec in self._tool_specs.values():
            function_payload = spec.definition.get("function")
            if not isinstance(function_payload, dict):
                continue
            name = function_payload.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            payload.append(
                {
                    "name": name.strip(),
                    "description": function_payload.get("description") or "",
                    "parameters": function_payload.get("parameters") or {"type": "object"},
                }
            )
        return payload

    def _format_component_inventory(
        self,
        components: list[ComponentItem],
        *,
        default_description: str,
    ) -> str:
        if not components:
            return "- none"
        lines: list[str] = []
        for item in components:
            description = self._component_description(item, default_description=default_description)
            lines.append(f"- {item.name}: {description}")
        return "\n".join(lines)

    def _format_local_tool_inventory(self) -> str:
        if not self.catalog.tools:
            return "- none"
        lines: list[str] = []
        for item in self.catalog.tools:
            description = self._component_description(
                item,
                default_description="Local tool wrapper callable from provider runtime.",
            )
            call_name = _safe_tool_name(f"aid_tool_{item.name}")
            lines.append(f"- {item.name} / {call_name}: {description}")
        return "\n".join(lines)

    def _component_description(self, item: ComponentItem, *, default_description: str) -> str:
        overrides = {
            "AID-skill-tool": (
                "Discover installed skills, read SKILL.md content, and run scripts from "
                "SKILLS/<skill>/scripts when extra knowledge is needed."
            ),
            "AID-tool-gateway": "Gateway bridge for MCP tool search/describe/call/health operations.",
        }
        if item.name in overrides:
            return overrides[item.name]
        description = (item.description or "").strip()
        return description or default_description

    def _stream_until_tool_call(self, messages: list[dict[str, Any]]) -> "_StreamResult":
        parser = _ToolCallStreamParser(allowed_tool_names=set(self._tool_specs.keys()))
        stream = self.provider.stream_chat(self.model, messages)
        try:
            for event in stream:
                for chunk in _extract_stream_text(event):
                    parser.feed(chunk)
                    if parser.tool_call is not None:
                        stop_stream = getattr(self.provider, "stop_stream", None)
                        if callable(stop_stream):
                            stop_stream()
                        return _StreamResult(text=parser.leading_text(), tool_call=parser.tool_call)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        return _StreamResult(text=parser.full_text(), tool_call=None)

    def _build_tool_specs(self) -> dict[str, ToolSpec]:
        specs: dict[str, ToolSpec] = {}

        # Generic skill helpers.
        specs["aid_list_skills"] = ToolSpec(
            definition=_fn_tool(
                name="aid_list_skills",
                description="List all installed skills with optional description.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            execute=lambda _args: [
                {
                    "name": item.name,
                    "path": str(item.path),
                    "description": item.description,
                }
                for item in self.catalog.skills
            ],
        )
        specs["aid_read_skill"] = ToolSpec(
            definition=_fn_tool(
                name="aid_read_skill",
                description="Read one skill markdown body from SKILLS/<skill>/SKILL.md.",
                parameters={
                    "type": "object",
                    "properties": {
                        "skill_name": {"type": "string", "description": "Skill folder name."},
                    },
                    "required": ["skill_name"],
                    "additionalProperties": False,
                },
            ),
            execute=self._tool_read_skill,
        )

        # MCP gateway bridge.
        specs["aid_mcp_gateway_call"] = ToolSpec(
            definition=_fn_tool(
                name="aid_mcp_gateway_call",
                description=(
                    "Call MCP gateway tools through scripts/gateway-call.mjs. "
                    "Useful tools: tool_search, tool_describe, tool_call, tool_health."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "description": "Gateway tool name."},
                        "payload": {"type": "object", "description": "Gateway tool JSON payload."},
                    },
                    "required": ["tool"],
                    "additionalProperties": False,
                },
            ),
            execute=self._tool_mcp_gateway_call,
        )

        # One callable function per local tool wrapper.
        for item in self.catalog.tools:
            tool_name = _safe_tool_name(f"aid_tool_{item.name}")
            specs[tool_name] = ToolSpec(
                definition=_fn_tool(
                    name=tool_name,
                    description=f"Run local tool wrapper {item.name}/tool.py with argv list.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "args": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Arguments passed to tool.py.",
                            },
                        },
                        "additionalProperties": False,
                    },
                ),
                execute=lambda args, item=item: self._tool_run_local_tool(item.name, args),
            )
        return specs

    def _tool_read_skill(self, args: dict[str, Any]) -> Any:
        skill_name = str(args.get("skill_name", "")).strip()
        if not skill_name:
            return {"error": "skill_name is required."}
        skill_file = self.repo_root / "SKILLS" / skill_name / "SKILL.md"
        if not skill_file.is_file():
            return {"error": f"Skill not found: {skill_name}"}
        return {
            "skill_name": skill_name,
            "content": _read_text(skill_file, fallback=""),
        }

    def _tool_mcp_gateway_call(self, args: dict[str, Any]) -> Any:
        tool = str(args.get("tool", "")).strip()
        if not tool:
            return {"error": "tool is required."}
        payload = args.get("payload")
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return {"error": "payload must be a JSON object."}

        script = self.repo_root / "MCP" / "AID-tool-gateway" / "scripts" / "gateway-call.mjs"
        if not script.is_file():
            return {"error": f"Gateway call script not found: {script}"}
        try:
            result = subprocess.run(
                ["node", str(script), "--tool", tool, "--payload", json.dumps(payload, ensure_ascii=False)],
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                check=False,
                timeout=MCP_GATEWAY_CALL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            return {
                "error": (
                    "Gateway call timed out "
                    f"after {MCP_GATEWAY_CALL_TIMEOUT_SECONDS} seconds."
                ),
                "stderr": (error.stderr or "").strip() if isinstance(error.stderr, str) else "",
            }
        if result.returncode != 0:
            return {
                "error": f"Gateway call failed with exit code {result.returncode}.",
                "stderr": result.stderr.strip(),
            }
        parsed = _parse_json_or_text(result.stdout)
        return parsed

    def _tool_run_local_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        script = self.repo_root / "TOOLS" / tool_name / "tool.py"
        if not script.is_file():
            return {"error": f"Tool script not found: {script}"}
        raw_args = args.get("args", [])
        argv: list[str] = []
        if isinstance(raw_args, list):
            for raw in raw_args:
                if isinstance(raw, str):
                    argv.append(raw)
        result = subprocess.run(
            [sys.executable, str(script), *argv],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        output = result.stdout.strip()
        parsed_output = _parse_json_or_text(output) if output else ""
        return {
            "exit_code": result.returncode,
            "stdout": parsed_output,
            "stderr": result.stderr.strip(),
        }


@dataclass(frozen=True)
class _StreamResult:
    text: str
    tool_call: dict[str, Any] | None


class _ToolCallStreamParser:
    def __init__(self, *, allowed_tool_names: set[str]) -> None:
        self._buffer = ""
        self._allowed_tool_names = set(allowed_tool_names)
        self._tool_call: dict[str, Any] | None = None

    @property
    def tool_call(self) -> dict[str, Any] | None:
        return self._tool_call

    def feed(self, chunk: str) -> None:
        if not chunk or self._tool_call is not None:
            return
        self._buffer += chunk
        self._tool_call = _try_extract_tool_call(self._buffer, allowed_tool_names=self._allowed_tool_names)

    def leading_text(self) -> str:
        if self._tool_call is None:
            return self._buffer
        marker_index = self._buffer.find("<AID_TOOL_CALL>")
        if marker_index < 0:
            return self._buffer
        return self._buffer[:marker_index]

    def full_text(self) -> str:
        return self._buffer


def _fn_tool(*, name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _safe_tool_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", value)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_") or "tool"


def _read_text(path: Path, *, fallback: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return fallback


def _parse_json_or_text(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _extract_response_text(payload: dict[str, Any]) -> str:
    # OpenAI-compatible: choices[0].message.content
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message")
        if isinstance(message, dict):
            text = _normalize_content_to_text(message.get("content"))
            if text:
                return text

    # Claude: content blocks at top-level
    content_blocks = payload.get("content")
    if isinstance(content_blocks, list):
        chunks: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks)

    # Gemini: candidates[].content.parts[].text
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            chunks = [part.get("text", "").strip() for part in parts if isinstance(part, dict)]
            chunks = [chunk for chunk in chunks if chunk]
            if chunks:
                return "\n".join(chunks)
    return ""


def _extract_stream_text(payload: dict[str, Any]) -> list[str]:
    chunks: list[str] = []

    # OpenAI-compatible: choices[].delta.content
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_text = part.get("text")
                    if isinstance(part_text, str):
                        chunks.append(part_text)

    # Claude: content_block_delta.delta.text
    event_type = payload.get("type")
    if event_type == "content_block_delta":
        delta = payload.get("delta")
        if isinstance(delta, dict):
            delta_text = delta.get("text")
            if isinstance(delta_text, str):
                chunks.append(delta_text)
    elif event_type == "content_block_start":
        content_block = payload.get("content_block")
        if isinstance(content_block, dict):
            block_text = content_block.get("text")
            if isinstance(block_text, str):
                chunks.append(block_text)

    # Gemini: candidates[].content.parts[].text
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)

    return chunks


def _try_extract_tool_call(buffer: str, *, allowed_tool_names: set[str] | None = None) -> dict[str, Any] | None:
    start = buffer.find("<AID_TOOL_CALL>")
    if start < 0:
        return None
    body_start = start + len("<AID_TOOL_CALL>")
    end = buffer.find("</AID_TOOL_CALL>", body_start)
    if end < 0:
        return None
    raw_json = buffer[body_start:end].strip()
    parsed = parse_json_object(raw_json)
    name = parsed.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    normalized_name = name.strip()
    if allowed_tool_names is not None and normalized_name not in allowed_tool_names:
        return None
    args = parse_json_object(parsed.get("arguments"))
    return {
        "name": normalized_name,
        "arguments": args,
        "raw_block": f"<AID_TOOL_CALL>\n{raw_json}\n</AID_TOOL_CALL>",
    }


def _normalize_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        return "\n".join(chunks).strip()
    return ""
