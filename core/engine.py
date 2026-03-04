"""Core runtime loop: inject context, run LLM, execute tools, persist outputs."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import platform
from pathlib import Path
import re
from typing import Any, Callable, Iterator, Protocol

from core.memory import MemoryStore
from core.models import ToolCall, ToolExecutionResult, TriggerEvent, TurnResult
from core.storage import JsonlStore
from core.tooling import ToolRegistry

TOOL_CALL_PATTERN = re.compile(
    r"<\s*tool_call\s*>\s*(\{.*?\})\s*<\s*/\s*tool_call\s*>",
    re.DOTALL | re.IGNORECASE,
)
TOOL_CALL_OPEN_PATTERN = re.compile(r"<\s*tool_call\s*>", re.IGNORECASE)
TOOL_CALL_CLOSE_PATTERN = re.compile(r"<\s*/\s*tool_call\s*>", re.IGNORECASE)
THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")
LEGACY_TOOL_CALL_PATTERN = re.compile(
    r"<\s*t\s*o\s*o\s*l\s*_\s*c\s*a\s*l\s*l\s*>(.*?)<\s*/\s*t\s*o\s*o\s*l\s*_\s*c\s*a\s*l\s*l\s*>",
    re.DOTALL | re.IGNORECASE,
)
LEGACY_ARG_KEY_OPEN_PATTERN = re.compile(
    r"<\s*a\s*r\s*g\s*_\s*k\s*e\s*y\s*>",
    re.IGNORECASE,
)
LEGACY_ARG_PAIR_PATTERN = re.compile(
    r"<\s*a\s*r\s*g\s*_\s*k\s*e\s*y\s*>(.*?)<\s*/\s*a\s*r\s*g\s*_\s*k\s*e\s*y\s*>"
    r"\s*"
    r"<\s*a\s*r\s*g\s*_\s*v\s*a\s*l\s*u\s*e\s*>(.*?)<\s*/\s*a\s*r\s*g\s*_\s*v\s*a\s*l\s*u\s*e\s*>",
    re.DOTALL | re.IGNORECASE,
)
LEGACY_ARG_TAG_PATTERN = re.compile(
    r"<\s*/?\s*a\s*r\s*g\s*_\s*[a-z_]+\s*>",
    re.IGNORECASE,
)
LEGACY_INLINE_CALL_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9_.:-]+)\s*(\{.*\})\s*$",
    re.DOTALL,
)


class LLMCompleter(Protocol):
    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...

    def complete_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]: ...

    def stop_stream(self) -> None: ...


class _ToolCallStreamFilter:
    """Streams assistant text while suppressing hidden XML blocks and emitting artifact chunks."""

    _BLOCKS = (
        ("<tool_call>", "</tool_call>"),
        ("<think>", "</think>"),
    )

    def __init__(self) -> None:
        self.buffer = ""
        self.in_block: tuple[str, str] | None = None
        self.current_artifact_id: str | None = None
        self.artifact_seq = 0
        self.visible_offset = 0

    def feed(
        self,
        chunk: str,
        *,
        on_artifact: Callable[[dict[str, Any]], None] | None = None,
        round_number: int,
    ) -> str:
        if not chunk:
            return ""
        self.buffer += chunk
        output: list[str] = []

        while self.buffer:
            if self.in_block is not None:
                _, end_tag = self.in_block
                end_idx = self.buffer.find(end_tag)
                if end_idx == -1:
                    if on_artifact is not None and self.current_artifact_id:
                        safe_len = len(self.buffer) - (len(end_tag) - 1)
                        if safe_len > 0:
                            part = self.buffer[:safe_len]
                            self.buffer = self.buffer[safe_len:]
                            on_artifact(
                                {
                                    "event": "chunk",
                                    "artifact_id": self.current_artifact_id,
                                    "round": round_number,
                                    "content": part,
                                }
                            )
                    return "".join(output)
                if end_idx > 0 and on_artifact is not None and self.current_artifact_id:
                    on_artifact(
                        {
                            "event": "chunk",
                            "artifact_id": self.current_artifact_id,
                            "round": round_number,
                            "content": self.buffer[:end_idx],
                        }
                    )
                if on_artifact is not None and self.current_artifact_id:
                    on_artifact(
                        {
                            "event": "end",
                            "artifact_id": self.current_artifact_id,
                            "round": round_number,
                        }
                    )
                self.buffer = self.buffer[end_idx + len(end_tag) :]
                self.in_block = None
                self.current_artifact_id = None
                continue

            found: tuple[int, tuple[str, str]] | None = None
            for start_tag, end_tag in self._BLOCKS:
                idx = self.buffer.find(start_tag)
                if idx == -1:
                    continue
                if found is None or idx < found[0]:
                    found = (idx, (start_tag, end_tag))
            if found is not None:
                start_idx, (start_tag, end_tag) = found
                if start_idx > 0:
                    visible_part = self.buffer[:start_idx]
                    output.append(visible_part)
                    self.visible_offset += len(visible_part)
                self.artifact_seq += 1
                artifact_id = f"r{round_number}-a{self.artifact_seq}"
                artifact_type = "think" if start_tag.lower() == "<think>" else "tool_call"
                if on_artifact is not None:
                    on_artifact(
                        {
                            "event": "start",
                            "artifact_id": artifact_id,
                            "type": artifact_type,
                            "round": round_number,
                            "offset": self.visible_offset,
                        }
                    )
                self.buffer = self.buffer[start_idx + len(start_tag) :]
                self.in_block = (start_tag, end_tag)
                self.current_artifact_id = artifact_id
                continue

            longest_start = max(len(start) for start, _ in self._BLOCKS)
            safe_len = len(self.buffer) - (longest_start - 1)
            if safe_len <= 0:
                return "".join(output)
            visible_part = self.buffer[:safe_len]
            output.append(visible_part)
            self.visible_offset += len(visible_part)
            self.buffer = self.buffer[safe_len:]

        return "".join(output)

    def finalize(self) -> str:
        if self.in_block is not None:
            self.buffer = ""
            return ""
        out = self.buffer
        self.buffer = ""
        return out


class AgentEngine:
    """Runs one event through the architecture shown in the diagram."""

    def __init__(
        self,
        *,
        repo_root: Path,
        llm: LLMCompleter,
        tools: ToolRegistry,
        history_store: JsonlStore,
        memory_store: MemoryStore,
        output_store: JsonlStore,
        system_prompt_override: str | None = None,
        history_enabled: bool = True,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.llm = llm
        self.tools = tools
        self.history_store = history_store
        self.memory_store = memory_store
        self.output_store = output_store
        self.system_prompt_override = (system_prompt_override or "").strip() or None
        self.history_enabled = history_enabled
        self._session_messages: list[dict[str, str]] = []

    def reset_session(self) -> None:
        self._session_messages.clear()

    def run_event(
        self,
        event: TriggerEvent,
        *,
        max_rounds: int = 6,
        on_stream: Callable[[str], None] | None = None,
        on_artifact: Callable[[dict[str, Any]], None] | None = None,
    ) -> TurnResult:
        used_tools: list[str] = []
        base_messages = self._build_initial_messages(event)
        messages = [dict(item) for item in base_messages]
        response_fragments: list[str] = []
        last_tool_signature: str | None = None
        repeated_same_tool_call = 0
        latest_tool_call_artifact_ids: dict[int, str] = {}

        def _emit_artifact(payload: dict[str, Any]) -> None:
            if on_artifact is None:
                return
            event_name = str(payload.get("event", "")).strip().lower()
            artifact_type = str(payload.get("type", "")).strip().lower()
            if event_name == "start" and artifact_type == "tool_call":
                artifact_id = str(payload.get("artifact_id", "")).strip()
                if artifact_id:
                    raw_round = payload.get("round")
                    try:
                        round_number = int(raw_round)
                    except (TypeError, ValueError):
                        round_number = 0
                    if round_number > 0:
                        latest_tool_call_artifact_ids[round_number] = artifact_id
            on_artifact(payload)

        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            assistant_text, emitted_artifacts_live = self._complete_assistant(
                messages,
                on_stream=on_stream,
                on_artifact=_emit_artifact if on_artifact is not None else None,
                round_number=rounds,
            )
            if not emitted_artifacts_live:
                self._emit_embedded_artifacts(
                    assistant_text,
                    on_artifact=_emit_artifact if on_artifact is not None else None,
                    round_number=rounds,
                )
            tool_call = self._extract_tool_call(assistant_text)

            if tool_call is None:
                visible = self._strip_think_blocks(assistant_text).strip()
                if visible:
                    response_fragments.append(visible)
                break

            tool_signature = self._tool_call_signature(tool_call)
            if tool_signature == last_tool_signature:
                repeated_same_tool_call += 1
            else:
                repeated_same_tool_call = 1
                last_tool_signature = tool_signature
            if repeated_same_tool_call >= 3:
                response_fragments.append(
                    "Stopped: repeated identical tool call detected. "
                    "Please revise the prompt or tool instructions."
                )
                break

            prelude = assistant_text.replace(tool_call.raw_block, "")
            prelude = self._strip_think_blocks(prelude).strip()
            if prelude:
                response_fragments.append(prelude)
            used_tools.append(tool_call.name)

            tool_result = self._execute_tool(tool_call)
            tool_result_payload = {
                "tool_name": tool_result.tool_name,
                "status": tool_result.status,
                "payload": tool_result.payload,
            }
            tool_result_json = json.dumps(tool_result_payload, ensure_ascii=False, indent=2)

            if on_artifact is not None:
                tool_artifact_id = latest_tool_call_artifact_ids.get(rounds)
                if tool_artifact_id:
                    _emit_artifact(
                        {
                            "event": "chunk",
                            "artifact_id": tool_artifact_id,
                            "round": rounds,
                            "content": "\n\nTool response (JSON):\n" + tool_result_json,
                        }
                    )

            messages.append({"role": "assistant", "content": assistant_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Tool response (JSON):\n"
                        + tool_result_json
                        + "\nContinue with your final user-facing answer."
                    ),
                }
            )

        if rounds >= max_rounds and not response_fragments:
            response_fragments.append(
                "Stopped: reached maximum reasoning rounds without a final answer."
            )

        final_response = "\n".join(item for item in response_fragments if item).strip()
        if not final_response:
            final_response = "No response generated."

        turn_result = TurnResult(
            event=event,
            response=final_response,
            rounds=rounds,
            used_tools=used_tools,
        )
        self._persist_turn(turn_result)
        self._append_session_turn(user_prompt=event.prompt, assistant_response=turn_result.response)
        return turn_result

    def _complete_assistant(
        self,
        messages: list[dict[str, Any]],
        *,
        on_stream: Callable[[str], None] | None,
        on_artifact: Callable[[dict[str, Any]], None] | None,
        round_number: int,
    ) -> tuple[str, bool]:
        if on_stream is None or not hasattr(self.llm, "complete_stream"):
            return self.llm.complete(messages, temperature=0.2), False

        assistant_text = ""
        stream_filter = _ToolCallStreamFilter()
        emitted_artifacts_live = False
        artifact_emitter = on_artifact
        if on_artifact is not None:
            def _emit_artifact(payload: dict[str, Any]) -> None:
                nonlocal emitted_artifacts_live
                emitted_artifacts_live = True
                on_artifact(payload)

            artifact_emitter = _emit_artifact
        snapshot_mode: bool | None = None
        last_raw_chunk = ""
        same_raw_streak = 0
        no_progress_streak = 0
        stream_guard_limit = 120
        for chunk in self.llm.complete_stream(messages, temperature=0.2):
            text = str(chunk)
            if not text:
                continue
            if text == last_raw_chunk:
                same_raw_streak += 1
            else:
                same_raw_streak = 0
            if snapshot_mode is None and last_raw_chunk:
                snapshot_mode = text.startswith(last_raw_chunk)
            if snapshot_mode:
                if text.startswith(last_raw_chunk):
                    delta = text[len(last_raw_chunk) :]
                else:
                    # Provider changed emission style mid-stream; recover safely.
                    delta = text
                assistant_text = text
            else:
                delta = text
                assistant_text += text

            last_raw_chunk = text
            visible = stream_filter.feed(
                delta,
                on_artifact=artifact_emitter,
                round_number=round_number,
            )
            if visible:
                on_stream(visible)
            if not delta and not visible:
                no_progress_streak += 1
            else:
                no_progress_streak = 0

            if same_raw_streak >= stream_guard_limit or no_progress_streak >= stream_guard_limit:
                if hasattr(self.llm, "stop_stream"):
                    try:
                        self.llm.stop_stream()
                    except Exception:  # noqa: BLE001
                        pass
                break

        tail = stream_filter.finalize()
        if tail:
            on_stream(tail)
        return assistant_text, emitted_artifacts_live

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        return THINK_PATTERN.sub("", text)

    @staticmethod
    def _emit_embedded_artifacts(
        assistant_text: str,
        *,
        on_artifact: Callable[[dict[str, Any]], None] | None,
        round_number: int,
    ) -> None:
        if on_artifact is None:
            return
        cursor = 0
        visible_offset = 0
        while cursor < len(assistant_text):
            think_match = THINK_PATTERN.search(assistant_text, cursor)
            tool_match = TOOL_CALL_PATTERN.search(assistant_text, cursor)

            next_match: tuple[str, re.Match[str]] | None = None
            if think_match is not None:
                next_match = ("think", think_match)
            if tool_match is not None and (
                next_match is None or tool_match.start() < next_match[1].start()
            ):
                next_match = ("tool_call", tool_match)
            if next_match is None:
                break

            artifact_type, match = next_match
            visible_offset += len(assistant_text[cursor : match.start()])
            if artifact_type == "think":
                content = match.group(1).strip()
            else:
                content = match.group(0).strip()
            if content:
                on_artifact(
                    {
                        "type": artifact_type,
                        "content": content,
                        "round": round_number,
                        "offset": visible_offset,
                    }
                )
            cursor = match.end()

    def _build_initial_messages(self, event: TriggerEvent) -> list[dict[str, str]]:
        system_prompt = self._build_system_message()
        tool_schemas = self.tools.schemas()

        injected_payload = {
            "tool_schemas": tool_schemas,
            "instructions": {
                "tool_call_format": "<tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>",
                "tool_policy": "When a tool is needed, output exactly one tool-call block and wait.",
            },
        }

        system_message = (
            f"{system_prompt}\n\n"
            "Internal runtime payload (not user prompt; do not discuss it unless asked):\n"
            + json.dumps(injected_payload, ensure_ascii=False, indent=2)
        )

        user_message = event.prompt.strip()

        return [
            {"role": "system", "content": system_message},
            *self._session_context_messages(),
            {"role": "user", "content": user_message},
        ]

    def _session_context_messages(self) -> list[dict[str, str]]:
        return [
            {"role": item["role"], "content": item["content"]}
            for item in self._session_messages
        ]

    def _append_session_turn(self, *, user_prompt: str, assistant_response: str) -> None:
        user_text = user_prompt.strip()
        assistant_text = assistant_response.strip()
        if user_text:
            self._session_messages.append({"role": "user", "content": user_text})
        if assistant_text:
            self._session_messages.append({"role": "assistant", "content": assistant_text})

    def _build_system_message(self) -> str:
        base_prompt = self._load_system_prompt()
        now_local = datetime.now().astimezone().replace(microsecond=0).isoformat()
        now_utc = datetime.now(UTC).replace(microsecond=0).isoformat()
        os_line = f"- OS: {platform.platform()}"
        linux_lines = self._linux_runtime_lines()
        runtime_rows = [
            "Runtime context:",
            os_line,
            *linux_lines,
            f"- Python: {platform.python_version()}",
            f"- Local time: {now_local}",
            f"- UTC time: {now_utc}",
        ]
        runtime_context = "\n".join(runtime_rows)
        return f"{base_prompt}\n\n{runtime_context}"

    def _linux_runtime_lines(self) -> list[str]:
        if platform.system().lower() != "linux":
            return []
        try:
            os_release = platform.freedesktop_os_release()
        except Exception:  # noqa: BLE001
            return ["- Linux distribution: unknown"]
        name = os_release.get("NAME", "").strip() or "unknown"
        version = os_release.get("VERSION_ID", "").strip() or os_release.get("VERSION", "").strip()
        kernel = platform.release()
        if version:
            distro = f"{name} {version}"
        else:
            distro = name
        return [
            f"- Linux distribution: {distro}",
            f"- Linux kernel: {kernel}",
        ]

    def _load_system_prompt(self) -> str:
        if self.system_prompt_override:
            return self.system_prompt_override

        custom_prompt_path = self.repo_root / "core" / "system_prompt.md"
        if custom_prompt_path.is_file():
            text = custom_prompt_path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text

        return (
            "You are the AIDZero runtime core.\n"
            "Architecture constraints:\n"
            "- Inputs come from a gateway (interactive).\n"
            "- Every turn includes system prompt + tool schemas.\n"
            "- Prior turns in the current app session are already included automatically.\n"
            "- Use history_get only when the user asks about turns from previous sessions/runs.\n"
            "- Memory is never injected automatically; use memory tools to read/write it.\n"
            "- If you need prior context, call memory_list and/or memory_get first.\n"
            "- You may call tools via <tool_call> JSON blocks.\n"
            "- Prefer short, actionable final answers.\n"
            "- If you modify memory, explain why in the final answer."
        )

    def _extract_tool_call(self, assistant_text: str) -> ToolCall | None:
        match = TOOL_CALL_PATTERN.search(assistant_text)
        if match is not None:
            raw_json = match.group(1)
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict):
                return None

            name = payload.get("name")
            arguments = payload.get("arguments", {})
            if not isinstance(name, str) or not name.strip():
                return None
            if not isinstance(arguments, dict):
                arguments = {"raw": arguments}
            return ToolCall(name=name.strip(), arguments=arguments, raw_block=match.group(0))

        unclosed = self._extract_unclosed_tool_call(assistant_text)
        if unclosed is not None:
            return unclosed

        legacy_match = LEGACY_TOOL_CALL_PATTERN.search(assistant_text)
        if legacy_match is None:
            return None

        body = legacy_match.group(1).strip()
        first_arg_match = LEGACY_ARG_KEY_OPEN_PATTERN.search(body)
        if first_arg_match is None:
            return self._parse_legacy_inline_tool_call(
                body=body,
                raw_block=legacy_match.group(0),
            )
        name_text = body[: first_arg_match.start()]
        name = " ".join(name_text.split()).strip()
        if not self._is_valid_tool_name(name):
            return None

        arguments: dict[str, Any] = {}
        for arg_match in LEGACY_ARG_PAIR_PATTERN.finditer(body):
            key = arg_match.group(1).strip()
            raw_value = arg_match.group(2).strip()
            if not key:
                continue
            arguments[key] = self._coerce_legacy_arg_value(raw_value)

        return ToolCall(name=name, arguments=arguments, raw_block=legacy_match.group(0))

    @staticmethod
    def _is_valid_tool_name(value: str) -> bool:
        return bool(value and TOOL_NAME_PATTERN.fullmatch(value))

    def _parse_legacy_inline_tool_call(self, *, body: str, raw_block: str) -> ToolCall | None:
        cleaned_body = LEGACY_ARG_TAG_PATTERN.sub("", body).strip()
        if not cleaned_body:
            return None

        inline_match = LEGACY_INLINE_CALL_PATTERN.match(cleaned_body)
        if inline_match is not None:
            name = inline_match.group(1).strip()
            if not self._is_valid_tool_name(name):
                return None
            raw_arguments = inline_match.group(2).strip()
            try:
                parsed_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed_arguments, dict):
                arguments: dict[str, Any] = parsed_arguments
            else:
                arguments = {"raw": parsed_arguments}
            return ToolCall(name=name, arguments=arguments, raw_block=raw_block)

        name = " ".join(cleaned_body.split()).strip()
        if not self._is_valid_tool_name(name):
            return None
        return ToolCall(name=name, arguments={}, raw_block=raw_block)

    def _extract_unclosed_tool_call(self, assistant_text: str) -> ToolCall | None:
        decoder = json.JSONDecoder()
        for open_match in TOOL_CALL_OPEN_PATTERN.finditer(assistant_text):
            payload_start = open_match.end()
            payload_text = assistant_text[payload_start:].lstrip()
            if not payload_text:
                continue
            leading_ws = len(assistant_text[payload_start:]) - len(payload_text)
            payload_start += leading_ws
            try:
                parsed_payload, offset = decoder.raw_decode(payload_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed_payload, dict):
                continue
            name = parsed_payload.get("name")
            arguments = parsed_payload.get("arguments", {})
            if not isinstance(name, str):
                continue
            name = name.strip()
            if not self._is_valid_tool_name(name):
                continue
            if not isinstance(arguments, dict):
                arguments = {"raw": arguments}

            payload_end = payload_start + offset
            tail = assistant_text[payload_end:]
            close_match = TOOL_CALL_CLOSE_PATTERN.match(tail.lstrip())
            raw_end = payload_end
            if close_match is not None:
                leading = len(tail) - len(tail.lstrip())
                raw_end = payload_end + leading + close_match.end()
            raw_block = assistant_text[open_match.start() : raw_end]
            return ToolCall(name=name, arguments=arguments, raw_block=raw_block)
        return None

    @staticmethod
    def _coerce_legacy_arg_value(raw_value: str) -> Any:
        if not raw_value:
            return ""
        looks_like_json = (
            raw_value[0] in {'{', '[', '"'}
            or raw_value in {"true", "false", "null"}
            or re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", raw_value) is not None
        )
        if looks_like_json:
            try:
                return json.loads(raw_value)
            except json.JSONDecodeError:
                return raw_value
        return raw_value

    @staticmethod
    def _tool_call_signature(tool_call: ToolCall) -> str:
        return json.dumps(
            {"name": tool_call.name, "arguments": tool_call.arguments},
            ensure_ascii=False,
            sort_keys=True,
        )

    def _execute_tool(self, tool_call: ToolCall) -> ToolExecutionResult:
        try:
            payload = self.tools.execute(tool_call.name, tool_call.arguments)
            return ToolExecutionResult(
                tool_name=tool_call.name,
                status="ok",
                payload=payload,
            )
        except Exception as error:  # noqa: BLE001
            return ToolExecutionResult(
                tool_name=tool_call.name,
                status="error",
                payload={"error": str(error)},
            )

    def _persist_turn(self, turn: TurnResult) -> None:
        now = datetime.now(UTC).replace(microsecond=0).isoformat()
        if self.history_enabled:
            history_row = {
                "timestamp": now,
                "event": {
                    "kind": turn.event.kind,
                    "source": turn.event.source,
                    "prompt": turn.event.prompt,
                    "metadata": turn.event.metadata,
                },
                "response": turn.response,
                "rounds": turn.rounds,
                "used_tools": turn.used_tools,
            }
            self.history_store.append(history_row)

        output_row = {
            "timestamp": now,
            "output": turn.response,
            "event_kind": turn.event.kind,
        }
        self.output_store.append(output_row)

        output_dir = self.repo_root / ".aidzero" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "latest.txt").write_text(turn.response + "\n", encoding="utf-8")
