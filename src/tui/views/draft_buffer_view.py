"""DraftBufferView - displays current drafts in the main view."""

from dataclasses import dataclass

from textual.message import Message
from textual.widgets import Static, ListView, ListItem
from textual.containers import Vertical

from ...core.draft_store import Draft
from ...core.session_v2 import SessionConfig


SIGNAL_THRESHOLD = 16  # Drafts <= this length are "signals"


def is_signal(draft: Draft) -> bool:
    """Check if a draft is a signal (short acknowledgment)."""
    return len(draft.text) <= SIGNAL_THRESHOLD


class DraftItem(ListItem):
    """A selectable list item representing a draft."""

    def __init__(self, draft: Draft, current_iter: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.draft = draft
        self._current_iter = current_iter

    def compose(self):
        """Compose the draft item content."""
        yield Static(self._format_content())

    def _format_content(self) -> str:
        """Format draft for display."""
        draft = self.draft
        age = self._current_iter - draft.iter
        draft_is_signal = is_signal(draft)

        lines = []

        # Header line
        seen_marker = "" if draft.seen else " [yellow]●[/]"
        type_label = "[cyan]signal[/]" if draft_is_signal else "draft"
        lines.append(f"[bold]#{draft.index}[/] {type_label} · age {age}{seen_marker}")

        # Text content
        if draft_is_signal:
            # Show signal inline, highlighted
            lines.append(f"  [bold cyan]{draft.text}[/]")
        else:
            # Show draft text
            text = draft.text.strip()
            # Indent multiline text
            indented = "\n".join("  " + line for line in text.split("\n"))
            lines.append(indented)

        return "\n".join(lines)


class DraftBufferView(Vertical):
    """
    Displays current draft responses with selection support.
    - Selectable list of drafts (arrow keys, mouse)
    - Each draft shows: index, age (relative), text, seen status
    - Visual distinction for short signals (<=16 chars)
    - Respects same display limits as mind sees
    """

    DEFAULT_CSS = """
    DraftBufferView {
        padding: 1;
    }

    DraftBufferView ListView {
        height: 1fr;
        background: transparent;
    }

    DraftBufferView ListItem {
        padding: 0 1;
        margin-bottom: 1;
    }

    DraftBufferView ListItem.--highlight {
        background: $primary 20%;
    }

    DraftBufferView .draft-signal {
        background: $surface;
    }

    DraftBufferView .draft-unseen {
        border-left: thick $warning;
    }

    DraftBufferView .empty-message {
        padding: 1;
        color: $text-muted;
    }
    """

    @dataclass
    class SelectionChanged(Message):
        """Posted when draft selection changes."""

        draft: Draft | None
        is_signal: bool

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
        self._list_view: ListView | None = None
        self._display_chars_used: int = 0

    def compose(self):
        """Compose the view."""
        yield ListView(id="draft-list")

    def on_mount(self) -> None:
        """Render initial state."""
        self._list_view = self.query_one("#draft-list", ListView)
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
        if self._list_view is None:
            return

        self._list_view.clear()

        if not self._drafts:
            # Show empty state (can't add non-ListItem to ListView, so use a placeholder)
            self._list_view.mount(
                ListItem(Static("[dim]No drafts yet[/]"), disabled=True)
            )
            return

        # Apply display limits (same as mind sees)
        visible = self._apply_limits(self._drafts)

        # Mount draft items (oldest first, newest at bottom)
        for draft in visible:
            item = DraftItem(draft, self._current_iter)
            # Add CSS classes for styling
            if is_signal(draft):
                item.add_class("draft-signal")
            if not draft.seen:
                item.add_class("draft-unseen")
            self._list_view.mount(item)

        # Select newest (last item) by default
        if visible:
            self._list_view.index = len(visible) - 1

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

        self._display_chars_used = total_chars

        # Reverse to get oldest first
        return list(reversed(result))

    @property
    def display_chars_used(self) -> int:
        """Return the number of chars currently being displayed."""
        return self._display_chars_used

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle selection change in the list."""
        if event.item is None:
            self.post_message(self.SelectionChanged(draft=None, is_signal=False))
            return

        if isinstance(event.item, DraftItem):
            draft = event.item.draft
            self.post_message(
                self.SelectionChanged(draft=draft, is_signal=is_signal(draft))
            )

    def get_selected_draft(self) -> Draft | None:
        """Get the currently selected draft, or None."""
        if self._list_view is None:
            return None

        highlighted = self._list_view.highlighted_child
        if highlighted is None:
            return None

        if isinstance(highlighted, DraftItem):
            return highlighted.draft

        return None
