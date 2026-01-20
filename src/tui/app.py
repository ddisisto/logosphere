"""
Mind TUI - Textual interface for Logosphere v2.

Phase 3: Single-step iteration with live updates.
"""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from textual.worker import Worker

from ..core.session_v2 import SessionV2
from ..core.lock import acquire_lock, release_lock, LockError
from ..mind import ops, MindRunner, MindConfig
from ..mind.events import EventType, IterationCompleteEvent
from .panels import IterationLog, StatusPanel
from .panels.iteration_log import IterationSummary
from .views import DraftBufferView, HistoryView
from .modals import UserModal


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
        Binding("s", "step_iteration", "Step"),
        Binding("h", "toggle_view", "History"),
        Binding("p", "cycle_presence", "Presence"),
        Binding("u", "cycle_user", "User"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, session_dir: Path | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session_dir = session_dir
        self._session: SessionV2 | None = None
        self._showing_history = False
        self._lock_held = False
        self._runner: MindRunner | None = None
        self._stepping = False  # Guard against double-stepping
        self._step_worker: Worker | None = None

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

            # Subscribe to runner events (for UI updates after step)
            self._runner.events.on(EventType.ITERATION_COMPLETE, self._on_iteration_complete)

            # Get or create active user and set to "reviewing" on launch
            active_user_id = self._session.user_registry.last_user_id
            if not active_user_id:
                active_user_id = "user"  # Default user
            ops.set_user_state(self._session, user_id=active_user_id, presence="reviewing")

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

        # State indicator
        if self._stepping:
            parts.append("[bold yellow]● STEPPING[/]")
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

    def action_cycle_presence(self) -> None:
        """Cycle presence state for active user."""
        if self._session is None:
            self.notify("No session loaded")
            return

        active_user_id = self._session.user_registry.last_user_id
        if not active_user_id:
            self.notify("No active user")
            return

        user = self._session.user_registry.get(active_user_id)
        if not user:
            self.notify("User not found")
            return

        # Cycle presence
        state = user.cycle_presence(self._session.iteration)
        self._session.save()

        # Update status panel
        self._refresh_status_panel()
        self.notify(f"{user.name}: {state.presence}")

    def action_cycle_user(self) -> None:
        """Open user management modal."""
        if self._session is None:
            self.notify("No session loaded")
            return

        def on_modal_dismiss(saved: bool) -> None:
            """Handle modal dismiss."""
            if saved:
                self._refresh_status_panel()
                self.notify("User settings saved")
            else:
                self._refresh_status_panel()  # Refresh anyway in case user switched

        self.push_screen(UserModal(self._session), on_modal_dismiss)

    def _refresh_status_panel(self) -> None:
        """Refresh the status panel with current data."""
        if self._session is None:
            return
        try:
            status = ops.get_session_status(self._session)
            status_panel = self.query_one("#status-panel", StatusPanel)
            status_panel.update_status(status)
        except Exception:
            pass

    def action_step_iteration(self) -> None:
        """Run a single iteration in the background."""
        if self._stepping:
            self.notify("Already stepping")
            return

        if self._session is None or self._runner is None:
            self.notify("No session loaded")
            return

        if not self._session.is_drafting:
            self.notify("Cannot step: session is idle (send a message first)")
            return

        self._stepping = True
        self._update_controls()

        # Run single iteration in background worker
        self._step_worker = self.run_worker(
            self._step_iteration_worker,
            thread=True,
            name="step-iteration",
        )

    def _step_iteration_worker(self) -> None:
        """Worker function that runs a single iteration in background thread."""
        if self._runner is None or self._session is None:
            self.call_from_thread(self._on_step_complete)
            return

        try:
            self._runner.step()
            # Event handler (_on_iteration_complete) will update UI
        except Exception as e:
            self.call_from_thread(self._handle_step_error, str(e))
        finally:
            self.call_from_thread(self._on_step_complete)

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

    def _on_step_complete(self) -> None:
        """Called when step completes (success or error)."""
        self._stepping = False
        self._update_controls()

    def _handle_step_error(self, error: str) -> None:
        """Handle step error on main thread."""
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
