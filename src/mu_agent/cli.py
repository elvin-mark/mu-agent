import asyncio
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

try:
    from .agent import Agent
    from .llm import get_provider
    from .permissions import PermissionManager, PermissionMode
    from .session import SessionManager
    from .types import Role
except ImportError:
    from mu_agent.agent import Agent
    from mu_agent.llm import get_provider
    from mu_agent.permissions import PermissionManager, PermissionMode
    from mu_agent.session import SessionManager
    from mu_agent.types import Role


class ToolConfirmationModal(ModalScreen[tuple[bool, bool]]):
    """Modal dialog for interactive tool approval."""

    def __init__(self, tool_name: str, args: dict[str, Any]):
        super().__init__()
        self.tool_name = tool_name
        self.args = args

    def compose(self) -> ComposeResult:
        arg_str = str(self.args)
        if len(arg_str) > 400:
            arg_str = arg_str[:400] + "..."
        yield Container(
            Label(
                "[bold yellow]⚠️  Tool Confirmation Requested[/bold yellow]",
                id="modal-title",
            ),
            Label(f"Tool: [bold cyan]{self.tool_name}[/bold cyan]"),
            Label(f"Arguments:\n[dim]{arg_str}[/dim]"),
            Horizontal(
                Button("Approve [Y]", id="btn-approve", variant="success"),
                Button("Reject [N]", id="btn-reject", variant="error"),
                Button("Always Approve [A]", id="btn-approve-all", variant="primary"),
                id="modal-buttons",
            ),
            id="modal-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-approve":
            self.dismiss((True, False))
        elif event.button.id == "btn-reject":
            self.dismiss((False, False))
        elif event.button.id == "btn-approve-all":
            self.dismiss((True, True))


class PiApp(App):
    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }
    #chat-log {
        height: 1fr;
        border: solid $accent;
        padding: 1;
        margin: 1;
        overflow-y: scroll;
        overflow-x: hidden;
    }
    #status-bar {
        background: $boost;
        color: $text;
        padding: 0 1;
    }
    #input-container {
        height: auto;
        padding: 0 1;
    }
    Input {
        width: 100%;
    }
    #modal-dialog {
        padding: 1 2;
        background: $panel;
        border: thick $accent;
        width: 70%;
        height: auto;
        align: center middle;
    }
    #modal-buttons {
        margin-top: 1;
        height: auto;
        align: center middle;
    }
    #modal-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear", "Clear", show=True),
    ]

    def __init__(
        self,
        provider_name: str = "openai",
        model: str = "gpt-4o",
        base_url: str | None = None,
        session_id: str | None = None,
        permission_mode: str = "ask",
    ):
        super().__init__()
        self.provider_name = provider_name
        self.model = model
        self.base_url = base_url
        self.session_manager = SessionManager(session_id=session_id)
        self.requested_permission_mode = permission_mode
        self.agent: Agent = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def request_tool_confirmation(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[bool, bool]:
        """Show interactive modal screen in TUI to request user tool approval."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[tuple[bool, bool]] = loop.create_future()

        def callback(result: tuple[bool, bool] | None):
            if result is None:
                result = (False, False)
            if not fut.done():
                fut.set_result(result)

        self.push_screen(ToolConfirmationModal(tool_name, args), callback)
        return await fut

    def on_mount(self) -> None:
        llm = get_provider(
            self.provider_name, default_model=self.model, base_url=self.base_url
        )
        perm_mode = PermissionMode(self.requested_permission_mode)
        perm_mgr = PermissionManager(
            mode=perm_mode, confirmation_callback=self.request_tool_confirmation
        )
        self.agent = Agent(
            llm=llm,
            session_manager=self.session_manager,
            permission_manager=perm_mgr,
        )
        asyncio.create_task(
            self.agent.mcp_manager.connect_and_register_tools(self.agent.tools)
        )
        log = self.query_one(RichLog)
        target = self.base_url or self.provider_name

        ascii_banner = r"""[bold cyan]
   __  _____  __  ___  __________  ________
  /  |/  / / / / / _ |/ ___/ __/ |/ /_  __/
 / /|_/ / /_/ / / __ / (_ / _//    / / /   
/_/  /_/\____/ /_/ |_\___/___/_/|_/ /_/    
[/bold cyan]"""

        log.write(ascii_banner)
        log.write(
            f"[bold green]⚡ Mu Agent v0.1.0[/bold green] | "
            f"Session: [bold yellow]{self.session_manager.session_id}[/bold yellow] | "
            f"Provider: [cyan]{self.provider_name}[/cyan] ({self.model}) @ [dim]{target}[/dim]\n"
        )

        # Display existing history if resuming
        if len(self.agent.messages) > 1:
            log.write("[dim]--- Resumed Session History ---[/dim]")
            for msg in self.agent.messages[1:]:
                if msg.role == Role.USER:
                    log.write(f"\n[bold yellow]User>[/bold yellow] {msg.content}")
                elif msg.role == Role.ASSISTANT and msg.content:
                    log.write(f"\n[bold blue]Mu>[/bold blue]\n{msg.content}")
            log.write("[dim]--------------------------------[/dim]\n")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
        yield Static("Tokens: 0 | Prompt: 0 | Completion: 0", id="status-bar")
        with Container(id="input-container"):
            yield Input(placeholder="Ask Mu anything... (Type /help for commands)")
        yield Footer()

    async def handle_slash_command(self, cmd_text: str, log: RichLog):
        parts = cmd_text.split()
        cmd = parts[0].lower()

        if cmd == "/stats":
            total = self.total_prompt_tokens + self.total_completion_tokens
            log.write(
                f"\n[bold green]📊 Telemetry & Token Usage Stats:[/bold green]\n"
                f"  - Prompt Tokens: {self.total_prompt_tokens}\n"
                f"  - Completion Tokens: {self.total_completion_tokens}\n"
                f"  - Total Session Tokens: {total}\n"
            )
        elif cmd == "/clear":
            log.clear()
            log.write("[dim]Console log cleared.[/dim]\n")
        elif cmd == "/session":
            log.write(
                f"\n[bold yellow]Session ID:[/bold yellow] {self.session_manager.session_id}\n"
                f"  - File: {self.session_manager.session_file}\n"
            )
        elif cmd == "/tools":
            log.write("\n[bold magenta]🛠️ Registered Agent Tools:[/bold magenta]")
            for schema in self.agent.tools.schemas:
                log.write(
                    f"  • [cyan]{schema['name']}[/cyan]: {schema.get('description', '')}"
                )
            log.write("")
        elif cmd == "/model":
            if len(parts) > 1:
                new_model = parts[1]
                self.model = new_model
                self.agent.llm.default_model = new_model
                log.write(
                    f"\n[bold green]Active model switched to:[/bold green] [cyan]{new_model}[/cyan]\n"
                )
            else:
                log.write(
                    f"\n[bold yellow]Active Model & Provider:[/bold yellow]\n"
                    f"  - Provider: [cyan]{self.provider_name}[/cyan]\n"
                    f"  - Model: [cyan]{self.model}[/cyan]\n"
                    f"  - Base URL: [dim]{self.base_url or 'default'}[/dim]\n"
                    f"  (Tip: Type [cyan]/model <name>[/cyan] to switch models at runtime)\n"
                )
        elif cmd == "/compact":
            old_len = len(self.agent.messages)
            self.agent.compact_context()
            log.write(
                "\n[bold green]🧹 Context Compaction Triggered:[/bold green] Truncated old tool logs to optimize token limits.\n"
            )
        elif cmd == "/system":
            log.write(
                f"\n[bold blue]⚙️ Active System Prompt:[/bold blue]\n[dim]{self.agent.system_prompt}[/dim]\n"
            )
        elif cmd == "/export":
            export_filename = f"pi_session_{self.session_manager.session_id}.md"
            try:
                with open(export_filename, "w", encoding="utf-8") as f:
                    f.write(
                        f"# Pi Session Export - {self.session_manager.session_id}\n\n"
                    )
                    for m in self.agent.messages:
                        f.write(f"### Role: {m.role.value}\n")
                        if m.content:
                            f.write(f"{m.content}\n\n")
                        if m.tool_calls:
                            f.writelines(
                                f"**Tool Call ({tc.name})**: `{tc.arguments}`\n\n"
                                for tc in m.tool_calls
                            )
                        if m.tool_result:
                            f.write(
                                f"**Tool Result ({m.tool_result.name})**:\n```\n{m.tool_result.output}\n```\n\n"
                            )
                log.write(
                    f"\n[bold green]📄 Session transcript exported to:[/bold green] [cyan]{export_filename}[/cyan]\n"
                )
            except Exception as err:
                log.write(f"\n[bold red]Error exporting session:[/bold red] {err!s}\n")
        elif cmd in ("/skills", "/skill"):
            if len(parts) > 1 and cmd == "/skill":
                skill_name = parts[1]
                if skill_name in self.agent.skill_manager.skills:
                    s = self.agent.skill_manager.skills[skill_name]
                    log.write(
                        f"\n[bold green]=== Skill: {s.name} ===[/bold green]\n"
                        f"Description: {s.description}\n"
                        f"File: [dim]{s.location}[/dim]\n\n"
                        f"--- Instructions ---\n{s.instructions}\n"
                    )
                else:
                    log.write(
                        f"\n[bold red]Error:[/bold red] Skill '{skill_name}' not found.\n"
                    )
            else:
                log.write("\n[bold magenta]✨ Discovered Skills:[/bold magenta]")
                if not self.agent.skill_manager.skills:
                    log.write(
                        "  [dim]No skills found in .mu/skills/ or ~/.mu/skills/[/dim]\n"
                    )
                else:
                    for s in self.agent.skill_manager.skills.values():
                        log.write(
                            f"  • [cyan]{s.name}[/cyan]: {s.description} ([dim]{s.location}[/dim])"
                        )
                    log.write(
                        "\n(Tip: Type [cyan]/skill <name>[/cyan] to view detailed instructions)\n"
                    )
        elif cmd == "/permission":
            if len(parts) > 1:
                target_mode = parts[1].lower()
                try:
                    mode_enum = PermissionMode(target_mode)
                    self.agent.permission_manager.set_mode(mode_enum)
                    log.write(
                        f"\n[bold green]🛡️  Permission mode updated to:[/bold green] [bold cyan]{mode_enum.value}[/bold cyan]\n"
                    )
                except ValueError:
                    log.write(
                        f"\n[bold red]Error:[/bold red] Invalid permission mode '{target_mode}'. Valid choices: yolo, ask, read_only\n"
                    )
            else:
                curr_mode = self.agent.permission_manager.mode.value
                auto_all = self.agent.permission_manager.session_approved_all
                log.write(
                    f"\n[bold green]🛡️  Current Permission Status:[/bold green]\n"
                    f"Mode: [bold cyan]{curr_mode}[/bold cyan]\n"
                    f"Session Auto-Approve: [cyan]{auto_all}[/cyan]\n"
                    f"\nUse [cyan]/permission <yolo|ask|read_only>[/cyan] to change mode.\n"
                )
        elif cmd == "/help":
            log.write(
                "\n[bold green]💡 Available Slash Commands:[/bold green]\n"
                "  • [cyan]/stats[/cyan]          - Display token usage & session telemetry\n"
                "  • [cyan]/clear[/cyan]          - Clear the terminal chat log\n"
                "  • [cyan]/permission [mode][/cyan] - Display or set permission mode (yolo, ask, read_only)\n"
                "  • [cyan]/session[/cyan]        - Display current session ID & file path\n"
                "  • [cyan]/tools[/cyan]          - List all available tools registered with Mu\n"
                "  • [cyan]/skills[/cyan]         - List all discovered skills in .mu/skills/\n"
                "  • [cyan]/skill [name][/cyan]   - Inspect instructions for a specific skill\n"
                "  • [cyan]/model [name][/cyan]   - View or switch active LLM model\n"
                "  • [cyan]/compact[/cyan]        - Manually trigger context compaction\n"
                "  • [cyan]/system[/cyan]         - View active system prompt & project rules\n"
                "  • [cyan]/export[/cyan]         - Export session transcript to Markdown file\n"
                "  • [cyan]/help[/cyan]           - Display this help message\n"
            )

        else:
            log.write(
                f"\n[bold red]Error:[/bold red] Unknown command '{cmd}'.\n"
                "Type [cyan]/help[/cyan] for a list of available commands.\n"
            )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return

        input_widget = self.query_one(Input)
        input_widget.value = ""
        log = self.query_one(RichLog)

        if user_text.startswith("/"):
            await self.handle_slash_command(user_text, log)
            return

        log.write(f"\n[bold yellow]User>[/bold yellow] {user_text}")
        self.agent.add_user_message(user_text)
        asyncio.create_task(self.run_agent_steps())

    async def run_agent_steps(self):
        log = self.query_one(RichLog)
        status_bar = self.query_one("#status-bar", Static)
        assistant_buf = ""

        try:
            async for event in self.agent.step():
                if event.type == "step_start":
                    log.write("\n[bold blue]Mu>[/bold blue]", scroll_end=True)
                    assistant_buf = ""

                elif event.type == "content_delta":
                    assistant_buf += event.payload
                    # Print completed words/lines in real-time while keeping formatting clean
                    if "\n" in event.payload or len(assistant_buf.split()) > 1:
                        # Flush completed text
                        lines = assistant_buf.split("\n")
                        if len(lines) > 1:
                            for line in lines[:-1]:
                                log.write(line, scroll_end=True)
                            assistant_buf = lines[-1]
                elif event.type == "usage":
                    u = event.payload
                    self.total_prompt_tokens += u.prompt_tokens
                    self.total_completion_tokens += u.completion_tokens
                    tot = self.total_prompt_tokens + self.total_completion_tokens
                    tok_per_sec = (
                        (u.completion_tokens / (u.latency_ms / 1000.0))
                        if u.latency_ms > 0
                        else 0
                    )
                    status_bar.update(
                        f"Tokens: [bold green]{tot}[/bold green] | "
                        f"Prompt: {self.total_prompt_tokens} | "
                        f"Completion: {self.total_completion_tokens} | "
                        f"Speed: {tok_per_sec:.1f} tok/s | "
                        f"Latency: {u.latency_ms:.0f}ms | "
                        f"Perms: [cyan]{self.agent.permission_manager.mode.value}[/cyan]"
                    )
                elif event.type == "step_complete":
                    if assistant_buf:
                        log.write(assistant_buf, scroll_end=True)
                        assistant_buf = ""
                elif event.type == "tool_call_start":
                    if assistant_buf:
                        log.write(assistant_buf, scroll_end=True)
                        assistant_buf = ""
                    tc = event.payload
                    log.write(
                        f"[bold magenta]⚡ Executing Tool:[/bold magenta] [cyan]{tc.name}[/cyan] with args: [dim]{tc.arguments}[/dim]"
                    )
                elif event.type == "tool_call_end":
                    out = event.payload["output"]
                    snippet = out[:300] + ("..." if len(out) > 300 else "")
                    log.write(f"[bold green]Result:[/bold green] [dim]{snippet}[/dim]")
        except Exception as err:
            log.write(f"\n[bold red]Error:[/bold red] {err!s}", scroll_end=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mu Agent CLI")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "ollama"],
        help="LLM Provider",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name (e.g. llama3.1, qwen2.5-coder, gpt-4o)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom base URL for OpenAI-compatible local server (e.g. http://localhost:11434/v1 or http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Resume an existing session ID from .mu/sessions/",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Run in autonomous YOLO mode (auto-approve tools except ultra-destructive operations)",
    )
    parser.add_argument(
        "--permission-mode",
        default="ask",
        choices=["yolo", "ask", "read_only"],
        help="Set tool execution permission mode (yolo, ask, read_only)",
    )

    args = parser.parse_args()
    mode = "yolo" if args.yolo else args.permission_mode

    app = PiApp(
        provider_name=args.provider,
        model=args.model,
        base_url=args.base_url,
        session_id=args.session,
        permission_mode=mode,
    )
    app.run()


if __name__ == "__main__":
    main()
