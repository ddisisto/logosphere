"""StatusPanel - displays session status in the sidebar."""

from textual.widgets import Static
from textual.app import ComposeResult

from ...mind.ops import SessionStatus


class StatusPanel(Static):
    """
    Displays session status:
    - Presence state (absent/reviewing/engaged)
    - Signal state (consecutive_hard / threshold)
    - Session metadata (iteration, pool sizes)
    - Awaiting message preview (if drafting)
    """

    DEFAULT_CSS = """
    StatusPanel {
        padding: 1;
    }
    """

    def __init__(self, status: SessionStatus, **kwargs) -> None:
        super().__init__(**kwargs)
        self._status = status

    def compose(self) -> ComposeResult:
        yield Static(self._render_content(), id="status-content")

    def update_status(self, status: SessionStatus) -> None:
        """Update with new status data."""
        self._status = status
        content = self.query_one("#status-content", Static)
        content.update(self._render_content())

    def _render_content(self) -> str:
        """Render the status panel content."""
        s = self._status
        lines = []

        # Users section
        presence_icons = {
            "absent": "[dim]○[/]",
            "reviewing": "[yellow]◐[/]",
            "engaged": "[green]●[/]",
        }

        if s.users:
            lines.append("[bold]Users:[/]")
            for user in s.users:
                is_active = user.id == s.active_user_id
                latest = user.get_latest_state()
                presence = latest.presence if latest else "absent"
                icon = presence_icons.get(presence, "?")
                marker = "▸ " if is_active else "  "
                name_style = "bold" if is_active else "dim"
                lines.append(f"{marker}{icon} [{name_style}]{user.name}[/]")
                if is_active and latest and latest.status:
                    lines.append(f"    [dim]{latest.status}[/]")
            lines.append("")
        else:
            # Fallback for no users
            icon = presence_icons.get(s.presence, "?")
            lines.append(f"{icon} [bold]{s.presence}[/]")
            if s.status_text:
                lines.append(f"  [dim]{s.status_text}[/]")
            lines.append("")

        # Signal state
        if s.signal_state:
            hard = s.signal_state.get("consecutive_hard", 0)
            thresh = s.signal_state.get("threshold", 3)
            lines.append(f"[bold]Signal:[/] {hard}/{thresh} hard")
        lines.append("")

        # Session metadata
        lines.append(f"[bold]Iteration:[/] {s.iteration}")
        lines.append(f"[bold]Thoughts:[/] {s.thoughts_active}/{s.thoughts_count}")
        lines.append(f"[bold]Model:[/] {s.model.split('/')[-1]}")
        lines.append("")

        # State-specific info
        if s.is_drafting:
            lines.append("[bold cyan]DRAFTING[/]")
            lines.append(f"Drafts: {s.drafts_count}")
            if s.awaiting_text:
                preview = s.awaiting_text[:50]
                if len(s.awaiting_text) > 50:
                    preview += "..."
                lines.append(f"[dim]Awaiting:[/]")
                lines.append(f"  {preview}")
        else:
            lines.append("[dim]IDLE[/]")
            lines.append(f"History: {s.history_count} entries")

        return "\n".join(lines)
