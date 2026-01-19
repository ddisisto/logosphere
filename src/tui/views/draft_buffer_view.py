"""DraftBufferView - displays current drafts in the main view."""

from textual.widgets import Static
from textual.containers import VerticalScroll

from ...core.dialogue_pool import Draft
from ...core.session_v2 import SessionConfig


class DraftBufferView(VerticalScroll):
    """
    Displays current draft responses.
    - Scrolling list of drafts, newest at bottom
    - Each draft shows: index, age (relative), text, seen status
    - Visual distinction for short signals (<=16 chars)
    - Respects same display limits as mind sees
    """

    DEFAULT_CSS = """
    DraftBufferView {
        padding: 1;
    }
    DraftBufferView > .draft-entry {
        margin-bottom: 1;
        padding: 0 1;
    }
    DraftBufferView > .draft-signal {
        background: $surface;
    }
    DraftBufferView > .draft-unseen {
        border-left: thick $warning;
    }
    """

    SIGNAL_THRESHOLD = 16  # Drafts <= this length are "signals"

    def __init__(
        self,
        drafts: list[Draft],
        current_iter: int,
        config: SessionConfig,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._drafts = drafts
        self._current_iter = current_iter
        self._config = config

    def on_mount(self) -> None:
        """Render initial state."""
        self._render_drafts()

    def update_drafts(
        self,
        drafts: list[Draft],
        current_iter: int,
    ) -> None:
        """Update with new draft data."""
        self._drafts = drafts
        self._current_iter = current_iter
        self._render_drafts()

    def _render_drafts(self) -> None:
        """Re-render all drafts."""
        self.remove_children()

        if not self._drafts:
            self.mount(Static("[dim]No drafts yet[/]", classes="draft-entry"))
            return

        # Apply display limits (same as mind sees)
        visible = self._apply_limits(self._drafts)

        # Show oldest first (bottom = newest for scroll)
        for draft in visible:
            self.mount(self._make_draft_widget(draft))

        # Scroll to bottom (newest)
        self.scroll_end(animate=False)

    def _apply_limits(self, drafts: list[Draft]) -> list[Draft]:
        """Apply display limits: max count and max chars."""
        max_count = self._config.draft_display_count
        max_chars = self._config.draft_display_chars

        # Select newest first up to limits
        result = []
        total_chars = 0
        for draft in reversed(drafts):
            if len(result) >= max_count:
                break
            if total_chars + len(draft.text) > max_chars and result:
                break
            result.append(draft)
            total_chars += len(draft.text)

        # Reverse to get oldest first
        return list(reversed(result))

    def _make_draft_widget(self, draft: Draft) -> Static:
        """Create widget for a single draft."""
        age = self._current_iter - draft.iter
        is_signal = len(draft.text) <= self.SIGNAL_THRESHOLD

        # Build CSS classes
        classes = ["draft-entry"]
        if is_signal:
            classes.append("draft-signal")
        if not draft.seen:
            classes.append("draft-unseen")

        # Format content
        content = self._format_draft(draft, age, is_signal)
        return Static(content, classes=" ".join(classes))

    def _format_draft(self, draft: Draft, age: int, is_signal: bool) -> str:
        """Format a single draft for display."""
        lines = []

        # Header line
        seen_marker = "" if draft.seen else " [yellow]●[/]"
        type_label = "[cyan]signal[/]" if is_signal else "draft"
        lines.append(f"[bold]#{draft.index}[/] {type_label} · age {age}{seen_marker}")

        # Text content
        if is_signal:
            # Show signal inline, highlighted
            lines.append(f"  [bold cyan]{draft.text}[/]")
        else:
            # Show draft text, possibly truncated for display
            text = draft.text.strip()
            # Indent multiline text
            indented = "\n".join("  " + line for line in text.split("\n"))
            lines.append(indented)

        return "\n".join(lines)
