"""Terminal UI application loop."""

from __future__ import annotations

from pathlib import Path

from agent.engine import AgentEngine
from agent.gateway import TriggerGateway
from agent.prompt_history import PromptHistoryStore


class TerminalApp:
    """Interactive and single-shot terminal interface."""

    def __init__(
        self,
        *,
        repo_root: Path,
        engine: AgentEngine,
        gateway: TriggerGateway,
        history: PromptHistoryStore,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.engine = engine
        self.gateway = gateway
        self.history = history

    def run(self, *, request: str | None, trigger: str) -> int:
        if request and request.strip():
            self.history.add_prompt(request)
            events = self.gateway.collect(trigger=trigger, prompt=request)
            return self._run_events(events)

        print("OpenClaw-style Terminal UI")
        print("Commands: /exit, /history, /heartbeat, /cron, /all")

        while True:
            try:
                user_input = self._prompt_with_history()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                return 0

            message = user_input.strip()
            if not message:
                continue
            if message in {"/exit", "/quit"}:
                return 0

            run_trigger = "interactive"
            run_prompt = message
            if message == "/history":
                self._print_history()
                continue
            if message == "/heartbeat":
                run_trigger = "heartbeat"
                run_prompt = ""
            elif message == "/cron":
                run_trigger = "cron"
                run_prompt = ""
            elif message == "/all":
                run_trigger = "all"
                run_prompt = ""

            if run_prompt:
                self.history.add_prompt(run_prompt)

            events = self.gateway.collect(trigger=run_trigger, prompt=run_prompt)
            if not events:
                print("No events available for that trigger.")
                continue
            self._run_events(events)

    def _prompt_with_history(self) -> str:
        recent = self.history.list_prompts(limit=8)
        if not recent:
            return input("\n> ")

        print("\nRecent prompts:")
        for index, item in enumerate(recent, start=1):
            print(f"{index}. {item}")
        choice = input("Select number or type a new prompt (Enter for new): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(recent):
                return recent[idx - 1]
            print("Invalid history index.")
            return ""
        if choice:
            return choice
        return input("> ")

    def _print_history(self) -> None:
        prompts = self.history.list_prompts(limit=30)
        if not prompts:
            print("History is empty.")
            return
        print("Prompt history:")
        for index, prompt in enumerate(prompts, start=1):
            print(f"{index}. {prompt}")

    def _run_events(self, events) -> int:
        for event in events:
            print(f"\n[{event.kind}] {event.source}")
            result = self.engine.run_event(event)
            print(result.response)
        return 0
