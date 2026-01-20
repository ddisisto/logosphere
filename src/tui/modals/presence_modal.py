"""Presence state modal."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from ...core.session_v2 import SessionV2
from ...core.users import User, PRESENCE_CYCLE


class PresenceModal(ModalScreen[bool]):
    """
    Modal for setting presence state and status.

    Returns True if saved, False if cancelled.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    CSS = """
    PresenceModal {
        align: center middle;
    }

    #modal-container {
        width: 60;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    #modal-title {
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }

    #user-label {
        text-align: center;
        color: $text-muted;
        padding-bottom: 1;
    }

    .field-row {
        height: 3;
        margin-bottom: 1;
    }

    .field-label {
        width: 12;
        padding-top: 1;
    }

    .field-input {
        width: 1fr;
    }

    #presence-select {
        width: 1fr;
    }

    #button-row {
        height: 3;
        margin-top: 1;
        align: center middle;
    }

    #button-row Button {
        margin: 0 1;
    }

    #hint {
        text-align: center;
        color: $text-muted;
        height: 1;
    }
    """

    def __init__(self, session: SessionV2, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session = session
        self._user: User | None = None

        # Load current user
        user_id = session.user_registry.last_user_id
        if user_id:
            self._user = session.user_registry.get(user_id)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label("Presence", id="modal-title")

            # Show which user
            user_name = self._user.name if self._user else "(no user)"
            yield Label(f"[dim]{user_name}[/]", id="user-label")

            # Presence selector
            with Horizontal(classes="field-row"):
                yield Label("Presence:", classes="field-label")
                yield Select(
                    [(p.capitalize(), p) for p in PRESENCE_CYCLE],
                    id="presence-select",
                    value=self._get_current_presence(),
                )

            # Status field
            with Horizontal(classes="field-row"):
                yield Label("Status:", classes="field-label")
                yield Input(
                    value=self._get_current_status(),
                    id="status-input",
                    classes="field-input",
                    placeholder="optional status text",
                )

            yield Static("[dim]Ctrl+S to save · Esc to cancel[/]", id="hint")

            # Buttons
            with Horizontal(id="button-row"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn", variant="default")

    def _get_current_presence(self) -> str:
        """Get current user's presence."""
        if self._user:
            latest = self._user.get_latest_state()
            if latest:
                return latest.presence
        return "absent"

    def _get_current_status(self) -> str:
        """Get current user's status."""
        if self._user:
            latest = self._user.get_latest_state()
            if latest:
                return latest.status
        return ""

    def action_save(self) -> None:
        """Save presence state."""
        if self._user is None:
            self.notify("No user selected", severity="error")
            self.dismiss(False)
            return

        # Get values
        presence = self.query_one("#presence-select", Select).value
        status = self.query_one("#status-input", Input).value.strip()

        # Update state (will replace if same iter)
        self._user.add_state(
            iter=self._session.iteration,
            presence=presence,
            status=status,
        )
        self._session.save()

        self.dismiss(True)

    def action_cancel(self) -> None:
        """Cancel and close modal."""
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save-btn":
            self.action_save()
        elif event.button.id == "cancel-btn":
            self.action_cancel()
