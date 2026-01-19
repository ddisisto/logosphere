"""
Mind TUI - Textual interface for Logosphere v2.

Phase 3: Iteration control with live updates.
"""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static
from textual.worker import Worker, WorkerState

from ..core.session_v2 import SessionV2
from ..core.lock import acquire_lock, release_lock, LockError
from ..mind import ops, MindRunner, MindConfig
from ..mind.events import EventType, IterationCompleteEvent, RunStopEvent
from .panels import IterationLog, StatusPanel
from .panels.iteration_log import IterationSummary
from .views import DraftBufferView, HistoryView


# Session tracking file (same as CLI)
SESSION_FILE = Path.home() / ".mind_session"


def get_current_session_dir() -> Path:
    """Get the current session directory."""
    if SESSION_FILE.exists():
        return Path(SESSION_FILE.read_text().strip())
    raise RuntimeError("No session open. Use 'mind open <dir>' first.")


class MindApp(App):
    """Main TUI application for Mind sessions."""

    TITLE = "Mind TUI"
    SUB_TITLE = "Logosphere v2"

    CSS = """
    #main-container {
        layout: horizontal;
    }

    #main-view {
        width: 3fr;
        border: solid $primary;
    }

    #sidebar {
        width: 1fr;
        layout: vertical;
    }

    #status-panel {
        height: auto;
        max-height: 50%;
        border: solid $secondary;
    }

    #iteration-log {
        height: 1fr;
        border: solid $secondary;
    }

    #controls {
        dock: bottom;
        height: 3;
        padding: 0 1;
        background: $surface;
    }

    #controls-content {
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True, show=False),
        Binding("r", "run_iterations", "Run"),
        Binding("p", "pause_iterations", "Pause"),
        Binding("h", "toggle_view", "Toggle history"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, session_dir: Path | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session_dir = session_dir
        self._session: SessionV2 | None = None
        self._showing_history = False
        self._lock_held = False
        self._runner: MindRunner | None = None
        self._running = False
        self._run_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="main-container"):
            # Main view area - starts with DraftBufferView, can toggle to HistoryView
            with Vertical(id="main-view"):
                # Placeholder - replaced on mount with actual data
                yield Static("Loading...", id="main-content")

            with Vertical(id="sidebar"):
                # Status panel (top of sidebar)
                yield Static("Loading...", id="status-panel")
                # Iteration log (bottom of sidebar)
                yield IterationLog(id="iteration-log")

        # Bottom controls
        with Horizontal(id="controls"):
            yield Static(self._render_controls(), id="controls-content")

        yield Footer()

    async def on_mount(self) -> None:
        """Called when app is mounted - load session and populate views."""
        try:
            # Determine session directory
            session_dir = self._session_dir or get_current_session_dir()
            self._session_dir = session_dir  # Store for later (lock release)

            # Acquire session lock
            try:
                acquire_lock(session_dir, "tui")
                self._lock_held = True
            except LockError as e:
                main = self.query_one("#main-content", Static)
                main.update(f"[bold red]Error:[/] {e}\n\nClose the other process or remove the lock file.")
                return

            # Load session
            self._session = SessionV2(session_dir)

            # Create runner for iterations
            config = MindConfig(verbose=False)
            self._runner = MindRunner(self._session, config)

            # Subscribe to runner events
            self._runner.events.on(EventType.ITERATION_COMPLETE, self._on_iteration_complete)
            self._runner.events.on(EventType.RUN_STOP, self._on_run_stop)

            # Set presence to "reviewing" on launch
            ops.set_signal(self._session, presence="reviewing")

            # Get session status
            status = ops.get_session_status(self._session)

            # Replace status panel placeholder (await ensures removal completes)
            status_panel_placeholder = self.query_one("#status-panel")
            await status_panel_placeholder.remove()
            await self.query_one("#sidebar").mount(
                StatusPanel(status, id="status-panel"),
                before=self.query_one("#iteration-log"),
            )

            # Replace main content with appropriate view
            await self._mount_main_view()

            # Update controls
            self._update_controls()

            # Update title with session path
            self.sub_title = f"{session_dir.name}"

        except RuntimeError as e:
            # Show error in main view
            main = self.query_one("#main-content", Static)
            main.update(f"[bold red]Error:[/] {e}")

    async def _mount_main_view(self) -> None:
        """Mount the appropriate main view based on session state."""
        if self._session is None:
            return

        main_container = self.query_one("#main-view")

        # Remove existing content (await each removal)
        for child in list(main_container.children):
            await child.remove()

        if self._showing_history:
            # Show history view
            history = self._session.get_history()
            view = HistoryView(
                history=history,
                current_iter=self._session.iteration,
                config=self._session.config,
                id="main-content",
            )
        else:
            # Show draft buffer view (default)
            drafts = self._session.get_all_drafts()
            view = DraftBufferView(
                drafts=drafts,
                current_iter=self._session.iteration,
                config=self._session.config,
                id="main-content",
            )

        await main_container.mount(view)

    def _update_controls(self) -> None:
        """Update the controls bar."""
        controls = self.query_one("#controls-content", Static)
        controls.update(self._render_controls())

    def _render_controls(self) -> str:
        """Render controls bar content."""
        if self._session is None:
            return "[dim]No session loaded[/]"

        parts = []

        # Run state indicator
        if self._running:
            parts.append("[bold green]● RUNNING[/]")
        elif self._session.is_drafting:
            parts.append("[cyan]DRAFTING[/]")
        else:
            parts.append("[dim]IDLE[/]")

        # View indicator
        view = "history" if self._showing_history else "drafts"
        parts.append(f"view: {view}")

        # Iteration counter
        parts.append(f"iter: {self._session.iteration}")

        return "  │  ".join(parts)

    async def action_toggle_view(self) -> None:
        """Toggle between draft buffer and history views."""
        self._showing_history = not self._showing_history
        await self._mount_main_view()
        self._update_controls()
        view_name = "history" if self._showing_history else "drafts"
        self.notify(f"Switched to {view_name} view")

    def action_run_iterations(self) -> None:
        """Start running iterations."""
        if self._running:
            self.notify("Already running")
            return

        if self._session is None or self._runner is None:
            self.notify("No session loaded")
            return

        if not self._session.is_drafting:
            self.notify("Cannot run: session is idle (send a message first)")
            return

        self._running = True
        self._update_controls()
        self.notify("Starting iterations...")

        # Run iterations in a background worker
        self._run_worker = self.run_worker(
            self._run_iterations_worker,
            thread=True,
            name="iteration-runner",
        )

    def action_pause_iterations(self) -> None:
        """Stop running iterations."""
        if not self._running:
            self.notify("Not running")
            return

        self._running = False
        self._update_controls()
        self.notify("Pausing...")

    def _run_iterations_worker(self) -> None:
        """Worker function that runs iterations in a background thread."""
        if self._runner is None or self._session is None:
            return

        # Run until stop condition or pause requested
        while self._running:
            try:
                # Run a single iteration
                result = self._runner.step()

                # Check stop conditions (runner events handle notification)
                if result.draft_added:
                    # Stop on draft in observe mode
                    self._running = False
                elif result.hard_signal and result.thoughts_added == 0:
                    # True silence - stop immediately
                    self._running = False
                elif result.hard_signal:
                    # Hard signal threshold check is in runner
                    pass  # Continue unless threshold reached

            except Exception as e:
                # Post error to main thread
                self.call_from_thread(self._handle_run_error, str(e))
                self._running = False
                break

    def _on_iteration_complete(self, event: IterationCompleteEvent) -> None:
        """Handle iteration complete event from runner (called from worker thread)."""
        # Create iteration summary for the log
        summary = IterationSummary(
            iteration=event.iteration,
            thoughts_count=event.thoughts_added,
            thoughts_chars=event.thoughts_chars,
            draft_added=event.draft_added,
            draft_length=event.draft_length,
            is_signal=event.draft_length is not None and event.draft_length <= 16,
        )

        # Post UI updates to main thread
        self.call_from_thread(self._update_after_iteration, summary, event)

    def _on_run_stop(self, event: RunStopEvent) -> None:
        """Handle run stop event from runner (called from worker thread)."""
        self._running = False
        self.call_from_thread(self._handle_run_stop, event)

    def _update_after_iteration(
        self, summary: IterationSummary, event: IterationCompleteEvent
    ) -> None:
        """Update UI after an iteration (called on main thread)."""
        # Update iteration log
        try:
            iteration_log = self.query_one("#iteration-log", IterationLog)
            iteration_log.add_entry(summary)
        except Exception:
            pass  # Widget might not exist yet

        # Update status panel with new signal state
        if self._session:
            try:
                status = ops.get_session_status(self._session)
                # Update signal_state with actual values from event
                status.signal_state = event.signal_state
                status_panel = self.query_one("#status-panel", StatusPanel)
                status_panel.update_status(status)
            except Exception:
                pass

        # Refresh main view to show new drafts
        if not self._showing_history:
            self._refresh_draft_view()

        # Update controls (iteration counter)
        self._update_controls()

    def _refresh_draft_view(self) -> None:
        """Refresh the draft buffer view with current data."""
        if self._session is None:
            return

        try:
            view = self.query_one("#main-content", DraftBufferView)
            view.update_drafts(
                drafts=self._session.get_all_drafts(),
                current_iter=self._session.iteration,
            )
        except Exception:
            pass  # View might be HistoryView or not mounted

    def _handle_run_stop(self, event: RunStopEvent) -> None:
        """Handle run stop on main thread."""
        self._running = False
        self._update_controls()
        self.notify(f"Stopped: {event.reason} ({event.iterations_run} iterations)")

    def _handle_run_error(self, error: str) -> None:
        """Handle run error on main thread."""
        self._running = False
        self._update_controls()
        self.notify(f"Error: {error}", severity="error")

    def action_quit(self) -> None:
        """Quit the application."""
        self._release_lock()
        self.exit()

    def _release_lock(self) -> None:
        """Release session lock if held."""
        if self._lock_held and self._session_dir:
            release_lock(self._session_dir)
            self._lock_held = False

    def on_unmount(self) -> None:
        """Called when app is unmounted - ensure lock is released."""
        self._release_lock()


def run_tui(session_dir: Path | None = None) -> None:
    """Run the Mind TUI."""
    app = MindApp(session_dir)
    app.run()
