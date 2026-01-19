"""HistoryView - displays conversation history in the main view."""

from textual.widgets import Static
from textual.containers import VerticalScroll

from ...core.dialogue_pool import HistoryEntry
from ...core.session_v2 import SessionConfig


class HistoryView(VerticalScroll):
    """
    Displays conversation history (alternate view, toggle via keybinding).
    - Shows user messages and accepted mind responses
    - Respects same display limits as mind sees
    - Oldest at top, newest at bottom
    """

    DEFAULT_CSS = """
    HistoryView {
        padding: 1;
    }
    HistoryView > .history-entry {
        margin-bottom: 1;
        padding: 0 1;
    }
    HistoryView > .history-user {
        border-left: thick $primary;
    }
    HistoryView > .history-mind {
        border-left: thick $success;
    }
    """

    def __init__(
        self,
        history: list[HistoryEntry],
        current_iter: int,
        config: SessionConfig,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._history = history
        self._current_iter = current_iter
        self._config = config

    def on_mount(self) -> None:
        """Render initial state."""
        self._render_history()

    def update_history(
        self,
        history: list[HistoryEntry],
        current_iter: int,
    ) -> None:
        """Update with new history data."""
        self._history = history
        self._current_iter = current_iter
        self._render_history()

    def _render_history(self) -> None:
        """Re-render all history entries."""
        self.remove_children()

        if not self._history:
            self.mount(Static("[dim]No conversation history[/]", classes="history-entry"))
            return

        # Apply display limits (same as mind sees)
        visible = self._apply_limits(self._history)

        for entry in visible:
            self.mount(self._make_entry_widget(entry))

        # Scroll to bottom (newest)
        self.scroll_end(animate=False)

    def _apply_limits(self, history: list[HistoryEntry]) -> list[HistoryEntry]:
        """Apply display limits: max count and max chars."""
        max_count = self._config.history_display_count
        max_chars = self._config.history_display_chars

        # Take from end (most recent)
        if len(history) <= max_count:
            selected = list(history)
        else:
            selected = list(history[-max_count:])

        # Apply char limit (trim from start)
        total_chars = sum(len(e.text) for e in selected)
        while selected and total_chars > max_chars:
            removed = selected.pop(0)
            total_chars -= len(removed.text)

        return selected

    def _make_entry_widget(self, entry: HistoryEntry) -> Static:
        """Create widget for a single history entry."""
        age = self._current_iter - entry.iter
        classes = ["history-entry", f"history-{entry.role}"]
        content = self._format_entry(entry, age)
        return Static(content, classes=" ".join(classes))

    def _format_entry(self, entry: HistoryEntry, age: int) -> str:
        """Format a single history entry for display."""
        lines = []

        # Header
        role_label = "[bold blue]user[/]" if entry.role == "user" else "[bold green]mind[/]"
        lines.append(f"{role_label} · age {age}")

        # Text content
        text = entry.text.strip()
        indented = "\n".join("  " + line for line in text.split("\n"))
        lines.append(indented)

        return "\n".join(lines)
