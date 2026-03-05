"""Textual-based TUI with fixed bottom input and collapsible artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Collapsible, Footer, Markdown, OptionList, Static, TextArea

from CORE.agents import AgentProfile, AgentProfileManager
from CORE.dash_commands import DashCommandRegistry, build_default_dash_command_registry
from CORE.engine import AgentEngine
from CORE.gateway import TriggerGateway
from CORE.prompt_history import PromptHistoryStore
from CORE.tooling import build_default_tool_registry
from CORE.ui_runtime import profile_disabled_tools


@dataclass
class _PendingResponse:
    body: Vertical
    live_text: Markdown
    block: Vertical
    stream_buffer: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    tail_start: int = 0
    tail_widget: Markdown | None = None


class _PromptTextArea(TextArea):
    """Input widget with custom key behavior for prompt submission."""

    def _on_key(self, event: events.Key) -> None:
        if event.key in {"enter", "ctrl+m"}:
            event.prevent_default()
            event.stop()
            app = self.app
            if app is not None and hasattr(app, "_submit_input_text"):
                app._submit_input_text()
            return
        if event.key == "ctrl+j":
            event.prevent_default()
            event.stop()
            self.insert("\n")
            app = self.app
            if app is not None and hasattr(app, "_resize_input_to_content"):
                app._resize_input_to_content()
            return
        super()._on_key(event)


class _TextualTUI(App[int]):
    _INPUT_MIN_HEIGHT = 3
    _INPUT_MAX_HEIGHT = 12

    CSS = """
    Screen {
      layout: vertical;
      background: $surface-darken-1;
    }
    #status {
      height: 1;
      padding: 0 1;
      background: $accent-darken-2;
      color: $text;
      text-style: bold;
    }
    #messages {
      height: 1fr;
      padding: 0 1;
    }
    #input {
      dock: bottom;
      height: 3;
      margin: 0 1 1 1;
      background: $panel-darken-1;
    }
    #command-selector {
      dock: bottom;
      margin: 0 1 4 1;
      display: none;
      max-height: 12;
      background: $panel-darken-1;
    }
    #command-selector.-visible {
      display: block;
    }
    #history-selector {
      layer: overlay;
      width: 88%;
      height: auto;
      max-height: 16;
      margin: 0 0 6 0;
      align-horizontal: center;
      background: $panel-darken-1;
      padding: 0 1;
      display: none;
    }
    #history-selector.-visible {
      display: block;
    }
    .user-prompt {
      color: $success;
      background: $panel-darken-1;
      margin: 0;
      padding: 0 1;
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
      background: $panel;
      height: auto;
    }
    .resp-header {
      height: 1;
      color: $accent-lighten-1;
      margin: 0;
      padding: 0 1;
      background: $panel-darken-1;
    }
    .resp-body {
      margin: 0;
      padding: 0 1;
      color: $text;
    }
    .resp-footer {
      color: $warning-lighten-1;
      margin: 0;
      padding: 0 1;
      background: $panel-darken-1;
    }
    .system-line {
      color: $text-muted;
      background: $surface;
      margin: 0;
      padding: 0 1;
    }
    .artifact {
      margin: 0;
      padding: 0;
      background: $panel-darken-1;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+s", "stop_generation", "Stop"),
        ("ctrl+up", "scroll_messages_up", "Scroll Up"),
        ("ctrl+down", "scroll_messages_down", "Scroll Down"),
        ("ctrl+pageup", "scroll_messages_page_up", "Page Up"),
        ("ctrl+pagedown", "scroll_messages_page_down", "Page Down"),
        ("ctrl+home", "scroll_messages_top", "Top"),
        ("ctrl+end", "scroll_messages_bottom", "Bottom"),
    ]

    def __init__(
        self,
        *,
        repo_root: Path,
        engine: AgentEngine,
        gateway: TriggerGateway,
        history: PromptHistoryStore,
        agent_manager: AgentProfileManager,
        agent_profile: AgentProfile,
        trigger: str,
        initial_request: str | None = None,
    ) -> None:
        super().__init__()
        self.repo_root = repo_root.resolve()
        self.engine = engine
        self.gateway = gateway
        self.history = history
        self.agent_manager = agent_manager
        self.agent_profile = agent_profile
        self.active_trigger = trigger.strip().lower() or "interactive"
        self.initial_request = (initial_request or "").strip()
        self._next_response_id = 1
        self._responses: dict[int, _PendingResponse] = {}
        self._busy = False
        self._stop_requested = False
        self._command_matches: list[str] = []
        self._history_matches: list[str] = []
        self.dash_commands: DashCommandRegistry = build_default_dash_command_registry(
            self.repo_root,
            enabled_modules=self.agent_profile.enabled_dash_modules,
        )

    def compose(self) -> ComposeResult:
        yield Static("", id="status", markup=False)
        yield VerticalScroll(id="messages")
        yield OptionList(id="command-selector")
        yield OptionList(id="history-selector")
        yield _PromptTextArea(
            "",
            id="input",
            placeholder="Type a prompt, /new, /history, /exit",
            soft_wrap=True,
            show_line_numbers=False,
        )
        yield Footer()

    def on_mount(self) -> None:
        self._reset_engine_session()
        self._update_status()
        self._resize_input_to_content()
        self.query_one("#input", TextArea).focus()
        if self.initial_request:
            self._queue_prompt(self.initial_request)

    def _submit_input_text(self) -> None:
        input_widget = self.query_one("#input", TextArea)
        raw = input_widget.text.strip()
        completed = self._resolve_completion_from_selector(raw)
        if completed != raw:
            # Enter with visible autocomplete should only complete the command.
            input_widget.text = completed
            self._move_cursor_to_input_end()
            self._resize_input_to_content()
            input_widget.focus()
            self._hide_command_selector()
            return
        raw = completed
        input_widget.clear()
        self._resize_input_to_content()
        self._hide_command_selector()
        if not raw:
            return
        if self._handle_command(raw):
            return
        self._queue_prompt(raw)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "input":
            return
        self._resize_input_to_content()
        self._refresh_command_selector(event.text_area.text)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_list_id = event.option_list.id or ""
        if option_list_id == "command-selector":
            completed = self._get_highlighted_completion()
            if not completed:
                return
            input_widget = self.query_one("#input", TextArea)
            input_widget.text = completed
            self._move_cursor_to_input_end()
            self._resize_input_to_content()
            input_widget.focus()
            self._hide_command_selector()
            return
        if option_list_id != "history-selector":
            return
        selected = self._get_highlighted_history_prompt()
        if selected is None:
            return
        input_widget = self.query_one("#input", TextArea)
        input_widget.text = selected
        self._resize_input_to_content()
        input_widget.focus()
        self._hide_history_selector()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape" and self._is_history_selector_visible():
            self._hide_history_selector()
            self.query_one("#input", TextArea).focus()
            event.stop()
            return

        input_widget = self.query_one("#input", TextArea)
        if self.focused is not input_widget:
            return
        if not self._command_matches:
            return
        selector = self.query_one("#command-selector", OptionList)
        if event.key == "down":
            selector.action_cursor_down()
            event.stop()
            return
        if event.key == "up":
            selector.action_cursor_up()
            event.stop()
            return
        if event.key == "tab":
            completed = self._get_highlighted_completion()
            if completed:
                input_widget.text = completed
                self._move_cursor_to_input_end()
                self._resize_input_to_content()
                self._hide_command_selector()
                event.stop()

    def action_stop_generation(self) -> None:
        if not self._busy:
            return
        self._stop_requested = True
        stop_stream = getattr(self.engine.llm, "stop_stream", None)
        if callable(stop_stream):
            try:
                stop_stream()
            except Exception:  # noqa: BLE001
                self._append_system_line("Stop request failed.")
                return
        self._append_system_line("Stop requested.")

    def on_paste(self, event: events.Paste) -> None:
        """Always route pasted text to the input field."""
        input_widget = self.query_one("#input", TextArea)
        input_widget.focus()
        input_widget.insert(event.text)
        self._resize_input_to_content()
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._scroll_messages_lines(-3)
        event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self._scroll_messages_lines(3)
        event.stop()

    def action_scroll_messages_up(self) -> None:
        self._scroll_messages_lines(-3)

    def action_scroll_messages_down(self) -> None:
        self._scroll_messages_lines(3)

    def action_scroll_messages_page_up(self) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        messages.scroll_page_up(animate=False)

    def action_scroll_messages_page_down(self) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        messages.scroll_page_down(animate=False)

    def action_scroll_messages_top(self) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        messages.scroll_home(animate=False)

    def action_scroll_messages_bottom(self) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        messages.scroll_end(animate=False)

    def _scroll_messages_lines(self, delta: int) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        messages.scroll_relative(y=delta, animate=False)

    def _resize_input_to_content(self) -> None:
        input_widget = self.query_one("#input", TextArea)
        text = input_widget.text
        line_count = max(1, text.count("\n") + 1)
        desired_height = line_count + 2
        clamped_height = max(self._INPUT_MIN_HEIGHT, min(self._INPUT_MAX_HEIGHT, desired_height))
        input_widget.styles.height = clamped_height

    def _set_input_text(self, text: str) -> None:
        input_widget = self.query_one("#input", TextArea)
        input_widget.text = text
        self._move_cursor_to_input_end()
        self._resize_input_to_content()
        input_widget.focus()
        self._hide_command_selector()

    def _move_cursor_to_input_end(self) -> None:
        input_widget = self.query_one("#input", TextArea)
        input_widget.cursor_location = input_widget.document.end

    def _handle_command(self, raw: str) -> bool:
        return self.dash_commands.handle(raw, app=self)

    def _append_system_line(self, text: str) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        row = Static(
            f"{datetime.now().strftime('%H:%M:%S')}  system  {text}",
            classes="system-line",
            markup=False,
        )
        messages.mount(row)
        messages.scroll_end(animate=False)

    def start_new_conversation(self) -> None:
        if self._busy:
            self._stop_requested = True
            stop_stream = getattr(self.engine.llm, "stop_stream", None)
            if callable(stop_stream):
                try:
                    stop_stream()
                except Exception:  # noqa: BLE001
                    pass

        self._responses.clear()
        self._next_response_id = 1
        self._hide_command_selector()
        self._hide_history_selector()

        messages = self.query_one("#messages", VerticalScroll)
        messages.remove_children()

        input_widget = self.query_one("#input", TextArea)
        input_widget.clear()
        self._resize_input_to_content()
        input_widget.focus()

        self._reset_engine_session()
        self._append_system_line("Started a new conversation.")

    def _reset_engine_session(self) -> None:
        reset = getattr(self.engine, "reset_session", None)
        if not callable(reset):
            return
        try:
            reset()
        except Exception as error:  # noqa: BLE001
            self._append_system_line(f"Session reset failed: {error}")

    def _update_status(self) -> None:
        status = self.query_one("#status", Static)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status.update(f"agent={self.agent_profile.name}  trigger=interactive  time={now}")

    def switch_agent_profile(self, profile_name: str) -> AgentProfile:
        profile = self.agent_manager.set_active_profile(profile_name)
        if bool(getattr(self.agent_manager, "is_remote", False)):
            self.agent_profile = profile
            self._hide_command_selector()
            self._update_status()
            return profile
        tools = build_default_tool_registry(
            self.repo_root,
            self.engine.memory_store,
            enabled_names=profile.enabled_tools,
            disabled_names=profile_disabled_tools(profile),
        )
        self.engine.tools = tools
        self.engine.system_prompt_override = profile.system_prompt
        self.engine.memory_store.enabled = profile.memory_enabled
        self.engine.history_enabled = profile.history_enabled
        self.history.enabled = profile.history_enabled
        self.dash_commands = build_default_dash_command_registry(
            self.repo_root,
            enabled_modules=profile.enabled_dash_modules,
        )
        self.agent_profile = profile
        self._hide_command_selector()
        self._update_status()
        return profile

    def _refresh_command_selector(self, raw_input: str) -> None:
        query = raw_input.strip().lower()
        if not query.startswith("/"):
            self._hide_command_selector()
            return
        matches = self.dash_commands.suggestions(query)
        if not matches:
            self._hide_command_selector()
            return
        selector = self.query_one("#command-selector", OptionList)
        selector.clear_options()
        for command, description in matches:
            selector.add_option(f"{command}  -  {description}")
        selector.highlighted = 0
        self._command_matches = [command for command, _ in matches]
        selector.add_class("-visible")

    def _hide_command_selector(self) -> None:
        selector = self.query_one("#command-selector", OptionList)
        selector.remove_class("-visible")
        self._command_matches = []

    def _is_command_selector_visible(self) -> bool:
        return self.query_one("#command-selector", OptionList).has_class("-visible")

    def _show_history_selector(self, *, limit: int = 30) -> bool:
        prompts = self.history.list_prompts(limit=limit)
        if not prompts:
            self._append_system_line("History is empty.")
            return False
        selector = self.query_one("#history-selector", OptionList)
        selector.clear_options()
        for idx, prompt in enumerate(prompts, start=1):
            one_line = " ".join(prompt.split())
            selector.add_option(f"{idx:>2}. {one_line}")
        selector.highlighted = 0
        self._history_matches = prompts
        selector.add_class("-visible")
        selector.focus()
        return True

    def _hide_history_selector(self) -> None:
        selector = self.query_one("#history-selector", OptionList)
        selector.remove_class("-visible")
        self._history_matches = []

    def _is_history_selector_visible(self) -> bool:
        return self.query_one("#history-selector", OptionList).has_class("-visible")

    def _get_highlighted_history_prompt(self) -> str | None:
        if not self._history_matches:
            return None
        selector = self.query_one("#history-selector", OptionList)
        highlighted = selector.highlighted
        if highlighted is None:
            return self._history_matches[0]
        if highlighted < 0 or highlighted >= len(self._history_matches):
            return None
        return self._history_matches[highlighted]

    def _get_highlighted_completion(self) -> str | None:
        if not self._command_matches:
            return None
        selector = self.query_one("#command-selector", OptionList)
        highlighted = selector.highlighted
        if highlighted is None:
            return self._command_matches[0]
        if highlighted < 0 or highlighted >= len(self._command_matches):
            return None
        return self._command_matches[highlighted]

    def _resolve_completion_from_selector(self, raw: str) -> str:
        if not raw.startswith("/"):
            return raw
        if not self._is_command_selector_visible():
            return raw
        if self.dash_commands.is_known_command(raw):
            return raw
        completed = self._get_highlighted_completion()
        return completed or raw

    def _queue_prompt(self, prompt: str) -> None:
        if self._busy:
            self._append_system_line("Still processing previous request. Please wait.")
            return
        self._append_user_prompt(prompt)
        self._busy = True
        self._stop_requested = False
        self.history.add_prompt(prompt)
        try:
            self.run_worker(
                lambda: self._process_prompt_worker(prompt),
                thread=True,
                group="prompt",
                exclusive=True,
            )
        except Exception as error:  # noqa: BLE001
            self._busy = False
            self._stop_requested = False
            self._append_system_line(f"Could not start request worker: {error}")

    def _process_prompt_worker(self, prompt: str) -> None:
        try:
            events = self.gateway.collect(trigger="interactive", prompt=prompt)
            if not events:
                self.call_from_thread(
                    self._append_system_line,
                    "No events available for current trigger.",
                )
                return

            for event in events:
                if self._stop_requested:
                    break
                response_id = self._next_response_id
                self._next_response_id += 1
                self.call_from_thread(self._create_response_block, response_id, event.kind, event.source)

                def _on_stream(chunk: str) -> None:
                    self.call_from_thread(self._append_stream, response_id, chunk)

                def _on_artifact(payload: dict[str, Any]) -> None:
                    self.call_from_thread(self._append_artifact, response_id, payload)

                try:
                    result = self.engine.run_event(event, on_stream=_on_stream, on_artifact=_on_artifact)
                except Exception as error:  # noqa: BLE001
                    self.call_from_thread(
                        self._finish_response,
                        response_id,
                        0,
                        0,
                        f"Provider/core error: {error}",
                    )
                    continue
                self.call_from_thread(
                    self._finish_response,
                    response_id,
                    result.rounds,
                    len(result.used_tools),
                    result.response,
                )
                if self._stop_requested:
                    break
        except Exception as error:  # noqa: BLE001
            try:
                self.call_from_thread(self._append_system_line, f"Unexpected worker error: {error}")
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._mark_idle_from_worker()

    def _mark_idle_from_worker(self) -> None:
        try:
            self.call_from_thread(self._mark_idle)
        except Exception:  # noqa: BLE001
            # Fallback used if the app is shutting down and thread handoff fails.
            self._busy = False
            self._stop_requested = False

    def _create_response_block(self, response_id: int, kind: str, source: str) -> None:
        messages = self.query_one("#messages", VerticalScroll)
        header = Static(
            f"{datetime.now().strftime('%H:%M:%S')}  assistant  kind={kind}  source={source}",
            classes="resp-header",
            markup=False,
        )
        live_text = Markdown("", classes="resp-body")
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
        footer = Static(
            f"stats  rounds={rounds}  tools={tools}",
            classes="resp-footer",
            markup=False,
        )
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
            if hasattr(content_widget, "update"):
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
                    pending.body.mount(Markdown(chunk, classes="resp-body"))
            artifact_type = str(entry["type"])
            round_number = entry.get("round")
            title = f"{artifact_type} · round {round_number}" if round_number else artifact_type
            content_widget = Markdown(str(entry["content"]), classes="resp-body")
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
        tail_widget = Markdown(tail, classes="resp-body")
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
        line = Static(
            f"{datetime.now().strftime('%H:%M:%S')}  you  {prompt}",
            classes="user-prompt",
            markup=False,
        )
        messages.mount(line)
        messages.scroll_end(animate=False)

    def _mark_idle(self) -> None:
        self._busy = False
        self._stop_requested = False
        self._update_status()
        self.query_one("#input", TextArea).focus()


def run_textual_tui(
    *,
    repo_root: Path,
    engine: AgentEngine,
    gateway: TriggerGateway,
    history: PromptHistoryStore,
    agent_manager: AgentProfileManager,
    agent_profile: AgentProfile,
    request: str | None,
    trigger: str,
) -> int:
    app = _TextualTUI(
        repo_root=repo_root,
        engine=engine,
        gateway=gateway,
        history=history,
        agent_manager=agent_manager,
        agent_profile=agent_profile,
        trigger=trigger,
        initial_request=request,
    )
    # Mouse is enabled by default so wheel scrolling works naturally in the TUI.
    # Set AIDZERO_TUI_MOUSE=0/false/no/off to disable it if you prefer terminal selection behavior.
    mouse_enabled = _read_bool_env("AIDZERO_TUI_MOUSE", default=True)
    result = app.run(mouse=mouse_enabled)
    return int(result or 0)


def _read_bool_env(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default
