"""
Mind TUI - Textual interface for Logosphere v2.

Phase 2: Basic shell with layout and static data.
"""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from ..core.session_v2 import SessionV2
from ..mind import ops
from .panels import IterationLog, StatusPanel
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
        Binding("h", "toggle_view", "Toggle history"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, session_dir: Path | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session_dir = session_dir
        self._session: SessionV2 | None = None
        self._showing_history = False

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
            # Load session
            session_dir = self._session_dir or get_current_session_dir()
            self._session = SessionV2(session_dir)

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

        # State indicator
        if self._session.is_drafting:
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

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def run_tui(session_dir: Path | None = None) -> None:
    """Run the Mind TUI."""
    app = MindApp(session_dir)
    app.run()
