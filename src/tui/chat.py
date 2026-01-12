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
from src.exchange import PREFIX_AUDIT, PREFIX_OBSERVER_ROLE, PREFIX_AUDITOR_ROLE
from src.exchange.auditor import invoke_auditor, format_audit_message


class StatusPanel(Static):
    """Displays session status."""

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.running = False
        self.progress = ""

    def refresh_status(self, view_mode: str = "audit", pool_target: int = 0) -> None:
        """Update status display."""
        status = self.session.get_status()

        # Running indicator with progress
        if self.running:
            indicator = f"[bold cyan]● thinking...[/]\n{self.progress}\n"
        else:
            indicator = "[bold green]● ready[/]\n"

        self.update(
            f"{indicator}"
            f"[bold]Pool:[/] {status['visible_messages']} messages\n"
            f"[bold]Active:[/] {status['active_pool_size']}\n"
            f"[bold]View:[/] {view_mode}\n"
        )

    def set_running(self, running: bool, progress: str = "") -> None:
        """Set running state."""
        self.running = running
        self.progress = progress


class PoolView(RichLog):
    """Displays pool messages with formatting."""

    # Message types to show in audit-only mode
    AUDIT_TYPES = {PREFIX_AUDIT, PREFIX_OBSERVER_ROLE, PREFIX_AUDITOR_ROLE}
    # User input is just ">>> text" (no [OBSERVER] prefix)

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
        # In audit-only mode, show audit-related messages and user input
        # User input is just ">>> text" (no prefix after >>>)
        if text.startswith(">>> "):
            after_marker = text[4:]
            # Show if it's a known audit type OR bare user input (no bracket prefix)
            if not after_marker.startswith("["):
                return True  # User input
            return any(after_marker.startswith(prefix) for prefix in self.AUDIT_TYPES)
        return any(text.startswith(prefix) for prefix in self.AUDIT_TYPES)

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
        elif display_text.startswith(PREFIX_OBSERVER_ROLE):
            style = "dim green"
            label = "OBSERVER ROLE"
            content = display_text[len(PREFIX_OBSERVER_ROLE):].strip()[:100] + "..."
        elif display_text.startswith(PREFIX_AUDITOR_ROLE):
            style = "dim cyan"
            label = "AUDITOR ROLE"
            content = display_text[len(PREFIX_AUDITOR_ROLE):].strip()[:100] + "..."
        elif text.startswith(">>> "):
            # User input is just ">>> text" (no [PREFIX] after >>>)
            if not display_text.startswith("["):
                style = "bold green"
                label = "YOU"
                content = display_text.strip()
            else:
                style = "yellow"
                label = "INJECT"
                content = display_text.strip()
        else:
            style = "white"
            label = "POOL"
            content = text.strip()

        # Truncate pool messages, but show AUDIT in full
        if label not in ("AUDIT", "YOU") and len(content) > 300:
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
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+f", "toggle_view", "Toggle full/audit view"),
        Binding("escape", "skip_or_clear", "Skip rotation / Clear input"),
    ]

    def __init__(self, session_dir: Path, **kwargs):
        super().__init__(**kwargs)
        self.session_dir = session_dir
        self.session: Optional[Session] = None
        self.runner: Optional[LogosRunner] = None
        self.config: Optional[LogosConfig] = None
        self.pool_size_at_last_input: int = 0  # Track for rotation-based triggering
        self._skip_rotation: bool = False  # Flag to skip remaining rotation

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="pool-container"):
                yield PoolView(self.session, id="pool-view")
            yield StatusPanel(self.session, id="status-panel")
        with Horizontal(id="input-container"):
            yield Input(placeholder="Type your response and press Enter", id="user-input")
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
        self.title = f"Logos Chat"
        self.sub_title = f"{self.session.current_branch}"

        # Initialize pool tracking - restore from session if available (crash recovery)
        saved_marker = self.session.config.get('chat_last_input_pool_size')
        if saved_marker is not None:
            self.pool_size_at_last_input = saved_marker
            thoughts_since = self.session.vector_db.size() - saved_marker
            self.notify(f"Restored state: {thoughts_since} thoughts since last input")
        else:
            self.pool_size_at_last_input = self.session.vector_db.size()

        # Auto-run to first audit on startup
        self.call_later(self._startup_run)

    def _startup_run(self) -> None:
        """On startup, continue rotation if incomplete, else ready for input."""
        current_size = self.session.vector_db.size()
        thoughts_since = current_size - self.pool_size_at_last_input

        # If we completed a cycle (pool size matches marker), ready for input
        if thoughts_since == 0 and self.session.config.get('chat_last_input_pool_size') is not None:
            self.notify("Ready for input")
            return

        # Otherwise, complete the rotation
        user_input = self.query_one("#user-input", Input)
        user_input.disabled = True

        if thoughts_since > 0:
            remaining = self.config.active_pool_size - thoughts_since
            self.notify(f"Resuming: {thoughts_since} thoughts done, {remaining} to go...")
        else:
            self.notify("Running initial rotation...")

        # Start rotation (callback will re-enable input when done)
        self.action_run_to_audit()

    def _refresh_status(self) -> None:
        """Refresh status panel with current view mode."""
        pool_view = self.query_one("#pool-view", PoolView)
        status_panel = self.query_one("#status-panel", StatusPanel)
        view_mode = "audit" if pool_view.audit_only else "full"
        status_panel.refresh_status(view_mode)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission - inject input, run rotation, then audit."""
        text = event.value.strip()
        if not text:
            return

        user_input = self.query_one("#user-input", Input)
        pool_view = self.query_one("#pool-view", PoolView)

        # Clear and disable input while running
        user_input.value = ""
        user_input.disabled = True

        # 1. Inject user input FIRST (so pool can respond to it)
        message = f">>> {text}"
        self.session.inject_message(
            text=message,
            embedding_client=self.runner.embedding_client,
            notes="chat input",
        )

        # Show injection immediately
        pool_view.refresh_new()
        self._refresh_status()

        # 2. Run rotation (pool generates thoughts, can see user input)
        # 3. Then audit cycle (AUDITOR ROLE, AUDIT, OBSERVER ROLE)
        # (callback will re-enable input when done)
        self.action_run_to_audit()

    def _thoughts_needed_for_rotation(self) -> int:
        """Compute thoughts needed to complete a full pool rotation."""
        current_size = self.session.vector_db.size()
        thoughts_since_input = current_size - self.pool_size_at_last_input
        needed = self.config.active_pool_size - thoughts_since_input
        return max(0, needed)

    def _run_single_step(self) -> bool:
        """Run a single iteration. Returns True if more steps needed."""
        if self._skip_rotation:
            return False

        current_size = self.session.vector_db.size()
        if current_size - self._rotation_start_size >= self._rotation_target:
            return False

        self.runner.step()
        self.session._save()
        return True

    def _start_rotation(self, target: int, start_size: int, callback) -> None:
        """Start rotation with periodic step execution."""
        self._rotation_target = target
        self._rotation_start_size = start_size
        self._rotation_callback = callback
        self._skip_rotation = False

        # Use set_interval for responsive event handling
        self._rotation_timer = self.set_interval(0.1, self._rotation_tick)

    def _rotation_tick(self) -> None:
        """Called periodically to run steps and update progress."""
        # Run a step (blocking but short)
        needs_more = self._run_single_step()

        # Update progress display
        self.session._load()
        current_size = self.session.vector_db.size()
        new_thoughts = current_size - self._rotation_start_size

        status_panel = self.query_one("#status-panel", StatusPanel)
        status_panel.set_running(True, f"+{new_thoughts}/{self._rotation_target} thoughts (Esc to skip)")
        self._refresh_status()

        if not needs_more:
            # Stop timer and proceed to callback
            self._rotation_timer.stop()
            self._rotation_callback()

    def action_run_to_audit(self) -> None:
        """Run until pool rotation complete, then invoke audit cycle."""
        target = self.config.active_pool_size
        start_size = self.pool_size_at_last_input

        self.notify(f"Running until {target} new thoughts...")

        # Start rotation with callback to finish
        self._start_rotation(target, start_size, self._finish_audit_cycle)

    def _finish_audit_cycle(self) -> None:
        """Complete the audit cycle after rotation."""
        status_panel = self.query_one("#status-panel", StatusPanel)
        pool_view = self.query_one("#pool-view", PoolView)

        if self._skip_rotation:
            self.notify("Skipped to audit...")

        # Invoke audit cycle directly
        status_panel.set_running(True, "auditor thinking...")
        self._refresh_status()

        # Run audit synchronously (it's an API call, will block briefly)
        self._invoke_audit_cycle_sync()

        # Update tracking for next rotation and persist for crash recovery
        self.pool_size_at_last_input = self.session.vector_db.size()
        self.session.config['chat_last_input_pool_size'] = self.pool_size_at_last_input
        self.session._save()

        # Done - clear running state
        status_panel.set_running(False)
        self.session._load()
        pool_view.refresh_new()
        self._refresh_status()
        self.notify("Your response?")

        # Re-enable input
        user_input = self.query_one("#user-input", Input)
        user_input.disabled = False
        user_input.focus()

    def _invoke_audit_cycle_sync(self) -> None:
        """Directly invoke audit cycle: AUDITOR ROLE → AUDIT → OBSERVER ROLE."""
        from pathlib import Path

        # Get prompts from config or defaults
        auditor_prompt = self.session.config.get('auditor_prompt', """You are the Auditor in a structured exchange protocol.

Your role:
- Summarize the current state of the Pool's reasoning
- Identify dominant themes, tensions, and emerging patterns
- Note any meta-level observations (self-reference, frame-shifts, requests)
- Your summary will be injected back into the Pool
- The Pool knows your instructions (this prompt is visible to them)

The Pool and Observer both read your summaries. Be concise but substantive.
Focus on what matters for continued productive reasoning.

Format: A single coherent summary, 2-4 paragraphs.
""")
        auditor_model = self.session.config.get('auditor_model', 'anthropic/claude-sonnet-4.5')

        # Load observer role
        observer_role_file = self.session.config.get('observer_role_file',
            Path(__file__).parent.parent.parent / ".daniel" / "ROLE.md")
        if isinstance(observer_role_file, str):
            observer_role_file = Path(observer_role_file)
        observer_role = observer_role_file.read_text().strip() if observer_role_file.exists() else "(Observer role not specified)"

        # 1. Inject AUDITOR ROLE
        auditor_role_msg = f"{PREFIX_AUDITOR_ROLE} {auditor_prompt}"
        self.session.inject_message(
            text=auditor_role_msg,
            embedding_client=self.runner.embedding_client,
            notes="chat audit cycle (auditor role)",
        )

        # 2. Get pool sample and invoke auditor
        sample_size = self.session.config.get('auditor_sample_size', 30)
        messages, _ = self.session.sample(sample_size)
        pool_prompt = self.config.system_prompt

        summary = invoke_auditor(
            pool_messages=messages,
            auditor_prompt=auditor_prompt,
            pool_prompt=pool_prompt,
            model=auditor_model,
        )

        # 3. Inject AUDIT summary
        audit_msg = format_audit_message(summary)
        self.session.inject_message(
            text=audit_msg,
            embedding_client=self.runner.embedding_client,
            notes="chat audit cycle (summary)",
        )

        # 4. Inject OBSERVER ROLE
        observer_role_msg = f"{PREFIX_OBSERVER_ROLE} {observer_role}"
        self.session.inject_message(
            text=observer_role_msg,
            embedding_client=self.runner.embedding_client,
            notes="chat audit cycle (observer role)",
        )

    def action_toggle_view(self) -> None:
        """Toggle between audit-only and full view."""
        pool_view = self.query_one("#pool-view", PoolView)
        is_audit = pool_view.toggle_view_mode()
        mode = "audit-only" if is_audit else "full"
        self.notify(f"View: {mode}")
        self._refresh_status()

    def action_skip_or_clear(self) -> None:
        """Skip rotation if running, otherwise clear input."""
        status_panel = self.query_one("#status-panel", StatusPanel)
        if status_panel.running:
            self._skip_rotation = True
            self.notify("Skipping rotation...")
        else:
            self.query_one("#user-input", Input).value = ""


def run_chat(session_dir: Path) -> None:
    """Run the chat TUI."""
    app = ChatApp(session_dir)
    app.run()
