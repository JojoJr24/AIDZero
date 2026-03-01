"""Core runtime loop: inject context, run LLM, execute tools, persist outputs."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any, Protocol

from agent.memory import MemoryStore
from agent.models import ToolCall, ToolExecutionResult, TriggerEvent, TurnResult
from agent.storage import JsonlStore
from agent.tooling import ToolRegistry

TOOL_CALL_PATTERN = re.compile(r"<AID_TOOL_CALL>\s*(\{.*?\})\s*</AID_TOOL_CALL>", re.DOTALL)


class LLMCompleter(Protocol):
    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...


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
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.llm = llm
        self.tools = tools
        self.history_store = history_store
        self.memory_store = memory_store
        self.output_store = output_store

    def run_event(self, event: TriggerEvent, *, max_rounds: int = 6) -> TurnResult:
        used_tools: list[str] = []
        base_messages = self._build_initial_messages(event)
        messages = [dict(item) for item in base_messages]
        response_fragments: list[str] = []

        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            assistant_text = self.llm.complete(messages, temperature=0.2)
            tool_call = self._extract_tool_call(assistant_text)

            if tool_call is None:
                response_fragments.append(assistant_text.strip())
                break

            prelude = assistant_text.replace(tool_call.raw_block, "").strip()
            if prelude:
                response_fragments.append(prelude)
            used_tools.append(tool_call.name)

            tool_result = self._execute_tool(tool_call)
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Tool response (JSON):\n"
                        + json.dumps(
                            {
                                "tool_name": tool_result.tool_name,
                                "status": tool_result.status,
                                "payload": tool_result.payload,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\nContinue with your final user-facing answer."
                    ),
                }
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
        return turn_result

    def _build_initial_messages(self, event: TriggerEvent) -> list[dict[str, str]]:
        system_prompt = self._load_system_prompt()
        tool_schemas = self.tools.schemas()
        history = self.history_store.tail(25)
        memory_snapshot = self.memory_store.all()

        injected_payload = {
            "event": {
                "kind": event.kind,
                "source": event.source,
                "created_at": event.created_at,
                "metadata": event.metadata,
            },
            "injected_every_turn": {
                "tool_schemas": tool_schemas,
                "jsonl_history": history,
                "memory": memory_snapshot,
            },
            "instructions": {
                "tool_call_format": "<AID_TOOL_CALL>{\"name\":\"tool_name\",\"arguments\":{}}</AID_TOOL_CALL>",
                "tool_policy": "When a tool is needed, output exactly one tool-call block and wait.",
            },
        }

        user_message = (
            "Gateway trigger received. Solve the task described below.\n\n"
            f"Trigger prompt:\n{event.prompt.strip()}\n\n"
            "Injected runtime payload:\n"
            + json.dumps(injected_payload, ensure_ascii=False, indent=2)
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _load_system_prompt(self) -> str:
        custom_prompt_path = self.repo_root / "agent" / "system_prompt.md"
        if custom_prompt_path.is_file():
            text = custom_prompt_path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text

        return (
            "You are OpenClaw-style runtime agent.\n"
            "Architecture constraints:\n"
            "- Inputs come from a gateway (heartbeat/cron/messengers/webhooks/interactive).\n"
            "- Every turn includes system prompt + tool schemas + JSONL history + memory.\n"
            "- You may call tools via <AID_TOOL_CALL> JSON blocks.\n"
            "- Prefer short, actionable final answers.\n"
            "- If you modify memory, explain why in the final answer."
        )

    def _extract_tool_call(self, assistant_text: str) -> ToolCall | None:
        match = TOOL_CALL_PATTERN.search(assistant_text)
        if match is None:
            return None

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
