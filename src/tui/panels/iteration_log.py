"""IterationLog - displays per-iteration summaries in the sidebar."""

from dataclasses import dataclass
from typing import Optional

from textual.widgets import Static
from textual.containers import VerticalScroll


@dataclass
class IterationSummary:
    """Summary of a single iteration."""
    iteration: int
    thoughts_count: int
    thoughts_chars: int
    draft_added: bool
    draft_length: Optional[int] = None
    is_signal: bool = False  # True if draft was <=16 chars (short signal)
    presence_change: Optional[str] = None  # e.g. "absent → reviewing"


class IterationLog(VerticalScroll):
    """
    Displays per-iteration summaries.
    - Shows last 3 iterations (configurable)
    - Latest/current at top
    - Each entry: iter#, thoughts, draft info, presence change
    """

    DEFAULT_CSS = """
    IterationLog {
        padding: 0 1;
    }
    IterationLog > Static {
        margin-bottom: 1;
    }
    """

    MAX_ENTRIES = 3

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries: list[IterationSummary] = []

    def on_mount(self) -> None:
        """Render initial state."""
        self._render_entries()

    def add_entry(self, entry: IterationSummary) -> None:
        """Add a new iteration entry (appears at top)."""
        self._entries.insert(0, entry)
        # Trim to max entries
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[:self.MAX_ENTRIES]
        self._render_entries()

    def update_current(self, entry: IterationSummary) -> None:
        """Update the current (top) entry - for live updates during iteration."""
        if self._entries:
            self._entries[0] = entry
        else:
            self._entries.append(entry)
        self._render_entries()

    def clear(self) -> None:
        """Clear all entries."""
        self._entries = []
        self._render_entries()

    def _render_entries(self) -> None:
        """Re-render all entries."""
        # Remove existing children
        self.remove_children()

        if not self._entries:
            self.mount(Static("[dim]No iterations yet[/]"))
            return

        for entry in self._entries:
            self.mount(Static(self._format_entry(entry)))

    def _format_entry(self, e: IterationSummary) -> str:
        """Format a single entry for display."""
        lines = [f"[bold]#{e.iteration}[/]"]

        # Thoughts
        lines.append(f"  Thoughts: {e.thoughts_count} ({e.thoughts_chars} chars)")

        # Draft
        if e.draft_added:
            if e.is_signal:
                lines.append(f"  Draft: [cyan]signal[/] ({e.draft_length} chars)")
            else:
                lines.append(f"  Draft: [green]yes[/] ({e.draft_length} chars)")
        else:
            lines.append("  Draft: [yellow]—[/]")

        # Presence change
        if e.presence_change:
            lines.append(f"  [dim]{e.presence_change}[/]")

        return "\n".join(lines)
