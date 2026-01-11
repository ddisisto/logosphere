"""
Logos Chat TUI - Interactive interface for structured exchange.

Uses textual for terminal UI. Designed with abstractions that
can later be adapted to web interface.
"""

from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, Input, RichLog
from textual.binding import Binding
from textual import work

from src.core.session import Session
from src.logos.config import LogosConfig
from src.logos.runner import LogosRunner
from src.exchange import PREFIX_AUDIT, PREFIX_OBSERVER, PREFIX_OBSERVER_ROLE, PREFIX_AUDITOR_ROLE


class StatusPanel(Static):
    """Displays session status."""

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.running = False

    def refresh_status(self, view_mode: str = "audit") -> None:
        """Update status display."""
        status = self.session.get_status()
        running_indicator = "[bold red]● RUNNING[/]\n" if self.running else ""
        self.update(
            f"{running_indicator}"
            f"[bold]Branch:[/] {status['current_branch']}\n"
            f"[bold]Iteration:[/] {status['iteration']}\n"
            f"[bold]Pool:[/] {status['visible_messages']} msgs\n"
            f"[bold]Active:[/] {status['active_pool_size']}\n"
            f"[bold]View:[/] {view_mode}\n"
        )

    def set_running(self, running: bool) -> None:
        """Set running state and refresh."""
        self.running = running


class PoolView(RichLog):
    """Displays pool messages with formatting."""

    # Message types to show in audit-only mode
    AUDIT_TYPES = {PREFIX_AUDIT, PREFIX_OBSERVER, PREFIX_OBSERVER_ROLE, PREFIX_AUDITOR_ROLE}

    def __init__(self, session: Session, **kwargs):
        super().__init__(highlight=True, markup=True, wrap=True, **kwargs)
        self.session = session
        self._last_vid = -1
        self.audit_only = True  # Default: show only audit-related messages

    def load_messages(self, limit: int = 50) -> None:
        """Load recent messages from pool."""
        self.clear()
        visible_ids = sorted(self.session.get_visible_ids())

        # Get last N messages
        recent_ids = visible_ids[-limit:] if len(visible_ids) > limit else visible_ids

        for vid in recent_ids:
            meta = self.session.vector_db.get_message(vid)
            if meta:
                self._display_message(vid, meta['text'])

        if recent_ids:
            self._last_vid = recent_ids[-1]

    def refresh_new(self) -> None:
        """Display any new messages since last refresh."""
        visible_ids = sorted(self.session.get_visible_ids())

        for vid in visible_ids:
            if vid > self._last_vid:
                meta = self.session.vector_db.get_message(vid)
                if meta:
                    self._display_message(vid, meta['text'])
                    self._last_vid = vid

    def toggle_view_mode(self) -> bool:
        """Toggle between audit-only and full view. Returns new mode."""
        self.audit_only = not self.audit_only
        self.load_messages()  # Reload with new filter
        return self.audit_only

    def _should_display(self, text: str) -> bool:
        """Check if message should be displayed based on current mode."""
        if not self.audit_only:
            return True
        # In audit-only mode, show audit-related messages
        # Check both with and without >>> prefix (external injection)
        return any(
            text.startswith(prefix) or text.startswith(f">>> {prefix}")
            for prefix in self.AUDIT_TYPES
        )

    def _display_message(self, vid: int, text: str) -> None:
        """Format and display a single message."""
        # Filter based on view mode
        if not self._should_display(text):
            return

        # Strip >>> prefix if present (external injection marker)
        display_text = text[4:] if text.startswith(">>> ") else text

        # Color-code by type
        if display_text.startswith(PREFIX_AUDIT):
            style = "bold cyan"
            label = "AUDIT"
            content = display_text[len(PREFIX_AUDIT):].strip()
        elif display_text.startswith(PREFIX_OBSERVER):
            style = "bold green"
            label = "YOU"
            content = display_text[len(PREFIX_OBSERVER):].strip()
        elif display_text.startswith(PREFIX_OBSERVER_ROLE):
            style = "dim green"
            label = "OBSERVER ROLE"
            content = display_text[len(PREFIX_OBSERVER_ROLE):].strip()[:100] + "..."
        elif display_text.startswith(PREFIX_AUDITOR_ROLE):
            style = "dim cyan"
            label = "AUDITOR ROLE"
            content = display_text[len(PREFIX_AUDITOR_ROLE):].strip()[:100] + "..."
        elif text.startswith(">>> "):
            style = "yellow"
            label = "INJECT"
            content = display_text.strip()
        else:
            style = "white"
            label = "POOL"
            content = text.strip()

        # Truncate long messages for display
        if len(content) > 300:
            content = content[:300] + "..."

        self.write(f"[{style}][{label}][/] {content}")


class ChatApp(App):
    """Main chat application."""

    CSS = """
    #main {
        layout: horizontal;
    }
    #pool-container {
        width: 3fr;
        border: solid green;
        padding: 0 1;
    }
    #status-panel {
        width: 1fr;
        border: solid blue;
        padding: 1;
    }
    #input-container {
        dock: bottom;
        height: 3;
        padding: 0 1;
    }
    #user-input {
        width: 100%;
    }
    PoolView {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+r", "run_rotation", "Run 1 rotation"),
        Binding("ctrl+s", "step", "Step 1 round"),
        Binding("ctrl+f", "toggle_view", "Toggle full/audit view"),
        Binding("escape", "clear_input", "Clear input"),
    ]

    def __init__(self, session_dir: Path, **kwargs):
        super().__init__(**kwargs)
        self.session_dir = session_dir
        self.session: Optional[Session] = None
        self.runner: Optional[LogosRunner] = None
        self.config: Optional[LogosConfig] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="pool-container"):
                yield PoolView(self.session, id="pool-view")
            yield StatusPanel(self.session, id="status-panel")
        with Horizontal(id="input-container"):
            yield Input(placeholder="Enter: send | Ctrl+R: rotate | Ctrl+S: step | Ctrl+F: toggle view", id="user-input")
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Load session
        self.session = Session(self.session_dir)
        self.config = LogosConfig.from_dict(self.session.config) if self.session.config else LogosConfig()
        self.runner = LogosRunner(self.session, self.config)

        # Update widgets with session
        pool_view = self.query_one("#pool-view", PoolView)
        pool_view.session = self.session
        pool_view.load_messages(50)

        status_panel = self.query_one("#status-panel", StatusPanel)
        status_panel.session = self.session
        self._refresh_status()

        # Set title
        self.title = f"Logos Chat - {self.session_dir.name}"
        self.sub_title = f"Branch: {self.session.current_branch}"

    def _refresh_status(self) -> None:
        """Refresh status panel with current view mode."""
        pool_view = self.query_one("#pool-view", PoolView)
        status_panel = self.query_one("#status-panel", StatusPanel)
        view_mode = "audit" if pool_view.audit_only else "full"
        status_panel.refresh_status(view_mode)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        text = event.value.strip()
        if not text:
            return

        # Clear input
        event.input.value = ""

        # Inject as observer response
        self._inject_observer(text)

    def _inject_observer(self, text: str) -> None:
        """Inject observer message and refresh."""
        message = f"{PREFIX_OBSERVER} {text}"
        self.session.inject_message(
            text=message,
            embedding_client=self.runner.embedding_client,
            notes="chat input",
        )

        # Refresh display
        pool_view = self.query_one("#pool-view", PoolView)
        pool_view.refresh_new()
        self._refresh_status()

    def _compute_rotation_rounds(self) -> int:
        """
        Compute rounds needed for 1 pool rotation.

        A rotation = enough iterations to cycle active_pool_size messages.
        Assumes ~2 messages per iteration on average.
        """
        active_size = self.session.active_pool_size
        messages_per_iter = 2  # Conservative estimate
        return max(10, active_size // messages_per_iter)

    @work(thread=True)
    def _run_iterations(self, count: int) -> None:
        """Run iterations in background thread."""
        self.runner.run(count)

    async def action_run_rotation(self) -> None:
        """Run 1 full rotation (active_pool_size messages worth)."""
        rounds = self._compute_rotation_rounds()
        self.notify(f"Running {rounds} rounds (1 rotation)...")

        # Show running indicator
        status_panel = self.query_one("#status-panel", StatusPanel)
        status_panel.set_running(True)
        self._refresh_status()

        self._run_iterations(rounds)

        # Wait and refresh
        await self._wait_and_refresh(rounds)

    async def action_step(self) -> None:
        """Run 1 round."""
        self.notify("Running 1 round...")

        status_panel = self.query_one("#status-panel", StatusPanel)
        status_panel.set_running(True)
        self._refresh_status()

        self._run_iterations(1)
        await self._wait_and_refresh(1)

    async def action_toggle_view(self) -> None:
        """Toggle between audit-only and full view."""
        pool_view = self.query_one("#pool-view", PoolView)
        is_audit = pool_view.toggle_view_mode()
        mode = "audit-only" if is_audit else "full"
        self.notify(f"View: {mode}")
        self._refresh_status()

    async def _wait_and_refresh(self, expected_rounds: int = 1) -> None:
        """Wait for runner and refresh display."""
        import asyncio
        # Wait proportional to expected rounds
        wait_time = min(60, 1.0 + expected_rounds * 0.5)
        await asyncio.sleep(wait_time)

        # Clear running indicator
        status_panel = self.query_one("#status-panel", StatusPanel)
        status_panel.set_running(False)

        # Reload session state
        self.session._load()

        pool_view = self.query_one("#pool-view", PoolView)
        pool_view.refresh_new()
        self._refresh_status()

        self.notify("Done.")

    def action_clear_input(self) -> None:
        """Clear input field."""
        self.query_one("#user-input", Input).value = ""


def run_chat(session_dir: Path) -> None:
    """Run the chat TUI."""
    app = ChatApp(session_dir)
    app.run()
