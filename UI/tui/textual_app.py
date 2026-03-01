"""Textual-based TUI with fixed bottom input and collapsible artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Collapsible, Footer, Input, Static

from agent.engine import AgentEngine
from agent.gateway import TriggerGateway
from agent.prompt_history import PromptHistoryStore


@dataclass
class _PendingResponse:
    body: Vertical
    live_text: Static
    block: Vertical
    stream_buffer: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    tail_start: int = 0
    tail_widget: Static | None = None


class _TextualTUI(App[int]):
    CSS = """
    Screen {
      layout: vertical;
    }
    #status {
      height: 1;
      padding: 0 1;
      background: $panel;
      color: $text-muted;
    }
    #messages {
      height: 1fr;
      padding: 0;
    }
    #input {
      dock: bottom;
      height: 3;
      margin: 0;
    }
    .user-prompt {
      color: $accent;
      margin: 0;
      padding: 0;
      height: auto;
    }
    .resp-body-wrap {
      margin: 0;
      padding: 0;
      height: auto;
    }
    .response-block {
      margin: 0;
      padding: 0;
      height: auto;
    }
    .resp-header {
      height: 1;
      color: $text-muted;
      margin: 0;
      padding: 0;
    }
    .resp-body {
      margin: 0;
      padding: 0;
    }
    .artifact {
      margin: 0;
      padding: 0;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        repo_root: Path,
        engine: AgentEngine,
        gateway: TriggerGateway,
        history: PromptHistoryStore,
        trigger: str,
        initial_request: str | None = None,
    ) -> None:
        super().__init__()
        self.repo_root = repo_root.resolve()
        self.engine = engine
        self.gateway = gateway
        self.history = history
        self.active_trigger = trigger.strip().lower() or "interactive"
        self.initial_request = (initial_request or "").strip()
        self._next_response_id = 1
        self._responses: dict[int, _PendingResponse] = {}
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Static("", id="status")
        yield VerticalScroll(id="messages")
        yield Input(placeholder="Type a prompt or /trigger <name>, /history, /exit", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self._update_status()
        if self.initial_request:
            self._queue_prompt(self.initial_request, trigger=self.active_trigger)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        event.input.value = ""
        if not raw:
            return
        if self._handle_command(raw):
            return
        self._queue_prompt(raw, trigger=self.active_trigger)

    def on_paste(self, event: events.Paste) -> None:
        """Always route pasted text to the input field."""
        input_widget = self.query_one("#input", Input)
        input_widget.focus()
        input_widget.insert_text_at_cursor(event.text)
        event.stop()

    def _handle_command(self, raw: str) -> bool:
        if raw in {"/exit", "/quit"}:
            self.exit(0)
            return True
        if raw == "/history":
            self._append_system_line("Recent prompts:")
            for idx, prompt in enumerate(self.history.list_prompts(limit=20), start=1):
                self._append_system_line(f"{idx:>2}. {prompt}")
            return True
        if raw.startswith("/trigger "):
            candidate = raw.removeprefix("/trigger ").strip().lower()
            valid = {"interactive", "heartbeat", "cron", "messengers", "webhooks", "all"}
            if candidate not in valid:
                self._append_system_line("Invalid trigger.")
                return True
            self.active_trigger = candidate
            self._update_status()
            self._append_system_line(f"Active trigger set to: {candidate}")
            return True
        return False

    def _append_system_line(self, text: str) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        row = Static(f"{datetime.now().strftime('%H:%M:%S')}  {text}", classes="resp-header")
        messages.mount(row)
        messages.scroll_end(animate=False)

    def _update_status(self) -> None:
        status = self.query_one("#status", Static)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status.update(f"trigger={self.active_trigger}  time={now}")

    def _queue_prompt(self, prompt: str, *, trigger: str) -> None:
        if self._busy:
            self._append_system_line("Still processing previous request. Please wait.")
            return
        self._append_user_prompt(prompt)
        self._busy = True
        self.history.add_prompt(prompt)
        self.run_worker(
            lambda: self._process_prompt_worker(prompt, trigger),
            thread=True,
            group="prompt",
            exclusive=True,
        )

    def _process_prompt_worker(self, prompt: str, trigger: str) -> None:
        events = self.gateway.collect(trigger=trigger, prompt=prompt)
        if not events:
            self.call_from_thread(self._append_system_line, "No events available for current trigger.")
            self.call_from_thread(self._mark_idle)
            return

        for event in events:
            response_id = self._next_response_id
            self._next_response_id += 1
            self.call_from_thread(self._create_response_block, response_id, event.kind, event.source)

            def _on_stream(chunk: str) -> None:
                self.call_from_thread(self._append_stream, response_id, chunk)

            def _on_artifact(payload: dict[str, Any]) -> None:
                self.call_from_thread(self._append_artifact, response_id, payload)

            result = self.engine.run_event(event, on_stream=_on_stream, on_artifact=_on_artifact)
            self.call_from_thread(
                self._finish_response,
                response_id,
                result.rounds,
                len(result.used_tools),
                result.response,
            )
        self.call_from_thread(self._mark_idle)

    def _create_response_block(self, response_id: int, kind: str, source: str) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        header = Static(f"{datetime.now().strftime('%H:%M:%S')}  {kind} · {source}", classes="resp-header")
        live_text = Static("", classes="resp-body")
        body = Vertical(live_text, classes="resp-body-wrap")
        block = Vertical(header, body, classes="response-block")
        messages.mount(block)
        self._responses[response_id] = _PendingResponse(
            body=body,
            live_text=live_text,
            block=block,
            tail_widget=live_text,
        )
        messages.scroll_end(animate=False)

    def _append_stream(self, response_id: int, chunk: str) -> None:
        if response_id not in self._responses:
            return
        pending = self._responses[response_id]
        pending.stream_buffer += chunk
        self._update_tail_widget(pending)
        self.query_one("#messages", VerticalScroll).scroll_end(animate=False)

    def _append_artifact(self, response_id: int, payload: dict[str, Any]) -> None:
        pending = self._responses.get(response_id)
        if pending is None:
            return
        event = str(payload.get("event", "")).strip().lower()
        if event:
            self._apply_artifact_event(pending, payload)
        else:
            # Backward-compatible single-shot artifact payload.
            artifact_type = str(payload.get("type", "artifact")).strip().lower() or "artifact"
            content = str(payload.get("content", "")).strip()
            if not content:
                return
            raw_offset = payload.get("offset")
            try:
                offset = int(raw_offset)
            except (TypeError, ValueError):
                offset = len(pending.stream_buffer)
            offset = max(0, min(offset, len(pending.stream_buffer)))
            pending.artifacts.append(
                {
                    "type": artifact_type,
                    "content": content,
                    "round": payload.get("round"),
                    "offset": offset,
                    "index": len(pending.artifacts),
                    "artifact_id": f"legacy-{len(pending.artifacts)}",
                    "open": False,
                    "content_widget": None,
                    "collapsible": None,
                }
            )
            self._rebuild_body(pending)
        self.query_one("#messages", VerticalScroll).scroll_end(animate=False)

    def _finish_response(self, response_id: int, rounds: int, tools: int, fallback_text: str) -> None:
        pending = self._responses.get(response_id)
        if pending is None:
            return
        if not pending.stream_buffer:
            pending.stream_buffer = fallback_text
        for artifact in pending.artifacts:
            artifact["open"] = False
            collapsible = artifact.get("collapsible")
            if isinstance(collapsible, Collapsible):
                collapsible.collapsed = True
        self._update_tail_widget(pending)
        footer = Static(f"rounds={rounds} tools={tools}", classes="resp-header")
        pending.block.mount(footer)
        self.query_one("#messages", VerticalScroll).scroll_end(animate=False)

    def _apply_artifact_event(self, pending: _PendingResponse, payload: dict[str, Any]) -> None:
        event = str(payload.get("event", "")).strip().lower()
        artifact_id = str(payload.get("artifact_id", "")).strip()
        if not artifact_id:
            return
        artifact = next((item for item in pending.artifacts if item.get("artifact_id") == artifact_id), None)
        if event == "start":
            artifact_type = str(payload.get("type", "artifact")).strip().lower() or "artifact"
            raw_offset = payload.get("offset")
            try:
                offset = int(raw_offset)
            except (TypeError, ValueError):
                offset = len(pending.stream_buffer)
            offset = max(0, min(offset, len(pending.stream_buffer)))
            pending.artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "type": artifact_type,
                    "content": "",
                    "round": payload.get("round"),
                    "offset": offset,
                    "index": len(pending.artifacts),
                    "open": True,
                    "content_widget": None,
                    "collapsible": None,
                }
            )
            self._rebuild_body(pending)
            return
        if artifact is None:
            return
        if event == "chunk":
            artifact["content"] = str(artifact.get("content", "")) + str(payload.get("content", ""))
            artifact["open"] = True
            content_widget = artifact.get("content_widget")
            if isinstance(content_widget, Static):
                content_widget.update(str(artifact["content"]))
            return
        if event == "end":
            artifact["open"] = True
            collapsible = artifact.get("collapsible")
            if isinstance(collapsible, Collapsible):
                collapsible.collapsed = False
            return

    def _rebuild_body(self, pending: _PendingResponse) -> None:
        entries = sorted(pending.artifacts, key=lambda item: (int(item["offset"]), int(item["index"])))
        pending.body.remove_children()
        cursor = 0
        for entry in entries:
            offset = int(entry["offset"])
            if offset > cursor:
                chunk = pending.stream_buffer[cursor:offset]
                if chunk:
                    pending.body.mount(Static(chunk, classes="resp-body"))
            artifact_type = str(entry["type"])
            round_number = entry.get("round")
            title = f"{artifact_type} · round {round_number}" if round_number else artifact_type
            content_widget = Static(str(entry["content"]), classes="resp-body")
            collapsible = Collapsible(
                content_widget,
                title=title,
                collapsed=not bool(entry.get("open", False)),
                classes="artifact",
            )
            entry["content_widget"] = content_widget
            entry["collapsible"] = collapsible
            pending.body.mount(collapsible)
            cursor = offset

        pending.tail_start = cursor
        tail = pending.stream_buffer[cursor:]
        tail_widget = Static(tail, classes="resp-body")
        pending.tail_widget = tail_widget
        pending.body.mount(tail_widget)

    def _update_tail_widget(self, pending: _PendingResponse) -> None:
        tail = pending.stream_buffer[pending.tail_start :]
        if pending.tail_widget is None:
            pending.live_text.update(tail)
            return
        pending.tail_widget.update(tail)

    def _append_user_prompt(self, prompt: str) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        line = Static(f"{datetime.now().strftime('%H:%M:%S')}  you> {prompt}", classes="user-prompt")
        messages.mount(line)
        messages.scroll_end(animate=False)

    def _mark_idle(self) -> None:
        self._busy = False
        self._update_status()
        self.query_one("#input", Input).focus()


def run_textual_tui(
    *,
    repo_root: Path,
    engine: AgentEngine,
    gateway: TriggerGateway,
    history: PromptHistoryStore,
    request: str | None,
    trigger: str,
) -> int:
    app = _TextualTUI(
        repo_root=repo_root,
        engine=engine,
        gateway=gateway,
        history=history,
        trigger=trigger,
        initial_request=request,
    )
    result = app.run()
    return int(result or 0)
