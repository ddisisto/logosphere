"""User management modal."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, Grid
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, Switch
from textual.message import Message

from ...core.session_v2 import SessionV2
from ...core.users import User, PresenceState, PRESENCE_CYCLE


class UserModal(ModalScreen[bool]):
    """
    Modal for user management.

    - Select/create users
    - Edit user settings (name, presence, status, show_clock, state_display_count)
    - View state history (read-only)
    - Pending changes shown until save
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    CSS = """
    UserModal {
        align: center middle;
    }

    #modal-container {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    #modal-title {
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }

    .field-row {
        height: 3;
        margin-bottom: 1;
    }

    .field-label {
        width: 18;
        padding-top: 1;
    }

    .field-input {
        width: 1fr;
    }

    #presence-select {
        width: 1fr;
    }

    #user-select {
        width: 1fr;
    }

    #state-history {
        height: auto;
        max-height: 12;
        border: solid $secondary;
        padding: 0 1;
        margin-top: 1;
    }

    #state-history-title {
        text-style: bold;
        padding: 1 0;
    }

    .state-entry {
        height: 1;
    }

    .pending-marker {
        color: $warning;
    }

    #button-row {
        height: 3;
        margin-top: 1;
        align: center middle;
    }

    #button-row Button {
        margin: 0 1;
    }

    #pending-indicator {
        color: $warning;
        text-align: center;
        height: 1;
    }
    """

    def __init__(self, session: SessionV2, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session = session
        self._current_user: User | None = None
        self._pending_changes: dict = {}

        # Load current user
        user_id = session.user_registry.last_user_id
        if user_id:
            self._current_user = session.user_registry.get(user_id)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label("User Management", id="modal-title")

            # User selector
            with Horizontal(classes="field-row"):
                yield Label("User:", classes="field-label")
                yield Select(
                    self._get_user_options(),
                    id="user-select",
                    value=self._current_user.id if self._current_user else None,
                )

            # Name field
            with Horizontal(classes="field-row"):
                yield Label("Name:", classes="field-label")
                yield Input(
                    value=self._current_user.name if self._current_user else "",
                    id="name-input",
                    classes="field-input",
                )

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

            # Show clock toggle
            with Horizontal(classes="field-row"):
                yield Label("Show clock:", classes="field-label")
                yield Switch(
                    value=self._current_user.show_clock if self._current_user else True,
                    id="show-clock-switch",
                )

            # State display count
            with Horizontal(classes="field-row"):
                yield Label("History count:", classes="field-label")
                yield Input(
                    value=str(self._current_user.state_display_count if self._current_user else 3),
                    id="display-count-input",
                    classes="field-input",
                    type="integer",
                )

            # Pending indicator
            yield Static("", id="pending-indicator")

            # State history (read-only)
            with Vertical(id="state-history"):
                yield Label("State History:", id="state-history-title")
                yield Static(self._render_state_history(), id="state-history-content")

            # Buttons
            with Horizontal(id="button-row"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("New User", id="new-user-btn", variant="default")
                yield Button("Cancel", id="cancel-btn", variant="default")

    def _get_user_options(self) -> list[tuple[str, str]]:
        """Get options for user selector."""
        users = self._session.user_registry.list_users()
        return [(u.name, u.id) for u in users] if users else [("(no users)", "")]

    def _get_current_presence(self) -> str:
        """Get current user's presence."""
        if self._current_user:
            latest = self._current_user.get_latest_state()
            if latest:
                return latest.presence
        return "absent"

    def _get_current_status(self) -> str:
        """Get current user's status."""
        if self._current_user:
            latest = self._current_user.get_latest_state()
            if latest:
                return latest.status
        return ""

    def _render_state_history(self) -> str:
        """Render state history for current user."""
        if not self._current_user:
            return "[dim]No user selected[/]"

        # Get pending count or current count
        count = self._pending_changes.get(
            'state_display_count',
            self._current_user.state_display_count
        )

        entries = self._current_user.get_state_for_display(self._session.iteration)

        # Show pending changes at top if any
        lines = []
        if self._pending_changes:
            pending_parts = []
            if 'presence' in self._pending_changes:
                pending_parts.append(f"presence: {self._pending_changes['presence']}")
            if 'status' in self._pending_changes:
                pending_parts.append(f"status: \"{self._pending_changes['status']}\"")
            if pending_parts:
                lines.append(f"[yellow]→ pending: {{{', '.join(pending_parts)}}}[/]")

        # Show history entries (limited by count)
        for i, (age, entry) in enumerate(entries[:count]):
            parts = [f"age: {age}", f"presence: {entry.presence}"]
            if entry.status:
                parts.append(f'status: "{entry.status}"')
            if entry.time:
                parts.append(f'time: "{entry.time}"')
            lines.append(f"[dim]{{{', '.join(parts)}}}[/]")

        if not entries and not self._pending_changes:
            lines.append("[dim]No state history[/]")

        return "\n".join(lines)

    def _update_pending_indicator(self) -> None:
        """Update the pending changes indicator."""
        indicator = self.query_one("#pending-indicator", Static)
        if self._pending_changes:
            indicator.update("[yellow]● Unsaved changes[/]")
        else:
            indicator.update("")

    def _update_state_history(self) -> None:
        """Refresh state history display."""
        content = self.query_one("#state-history-content", Static)
        content.update(self._render_state_history())

    def on_input_changed(self, event: Input.Changed) -> None:
        """Track changes to input fields."""
        if event.input.id == "name-input":
            current = self._current_user.name if self._current_user else ""
            if event.value != current:
                self._pending_changes['name'] = event.value
            elif 'name' in self._pending_changes:
                del self._pending_changes['name']

        elif event.input.id == "status-input":
            current = self._get_current_status()
            if event.value != current:
                self._pending_changes['status'] = event.value
            elif 'status' in self._pending_changes:
                del self._pending_changes['status']

        elif event.input.id == "display-count-input":
            try:
                new_count = int(event.value) if event.value else 3
                current = self._current_user.state_display_count if self._current_user else 3
                if new_count != current:
                    self._pending_changes['state_display_count'] = new_count
                elif 'state_display_count' in self._pending_changes:
                    del self._pending_changes['state_display_count']
                # Live update history display
                self._update_state_history()
            except ValueError:
                pass

        self._update_pending_indicator()
        self._update_state_history()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "user-select":
            # Switch to different user
            if event.value:
                self._switch_to_user(str(event.value))

        elif event.select.id == "presence-select":
            current = self._get_current_presence()
            if event.value != current:
                self._pending_changes['presence'] = event.value
            elif 'presence' in self._pending_changes:
                del self._pending_changes['presence']
            self._update_pending_indicator()
            self._update_state_history()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        if event.switch.id == "show-clock-switch":
            current = self._current_user.show_clock if self._current_user else True
            if event.value != current:
                self._pending_changes['show_clock'] = event.value
            elif 'show_clock' in self._pending_changes:
                del self._pending_changes['show_clock']
            self._update_pending_indicator()

    def _switch_to_user(self, user_id: str) -> None:
        """Switch to a different user."""
        if self._pending_changes:
            # Could prompt to save, for now just discard
            self._pending_changes.clear()

        self._current_user = self._session.user_registry.get(user_id)
        if self._current_user:
            self._session.user_registry.last_user_id = user_id
            self._refresh_fields()

    def _refresh_fields(self) -> None:
        """Refresh all fields with current user data."""
        if not self._current_user:
            return

        self.query_one("#name-input", Input).value = self._current_user.name
        self.query_one("#status-input", Input).value = self._get_current_status()
        self.query_one("#presence-select", Select).value = self._get_current_presence()
        self.query_one("#show-clock-switch", Switch).value = self._current_user.show_clock
        self.query_one("#display-count-input", Input).value = str(self._current_user.state_display_count)
        self._update_state_history()
        self._update_pending_indicator()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save-btn":
            self.action_save()
        elif event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "new-user-btn":
            self._create_new_user()

    def _create_new_user(self) -> None:
        """Create a new user."""
        # Generate unique ID
        existing_ids = set(self._session.user_registry.user_ids())
        base = "user"
        new_id = base
        counter = 1
        while new_id in existing_ids:
            counter += 1
            new_id = f"{base}{counter}"

        # Create user
        user = self._session.user_registry.create(new_id, new_id)
        self._session.user_registry.last_user_id = new_id
        self._session.save()

        # Update selector and switch to new user
        select = self.query_one("#user-select", Select)
        select.set_options(self._get_user_options())
        select.value = new_id

        self._current_user = user
        self._pending_changes.clear()
        self._refresh_fields()

    def action_save(self) -> None:
        """Save changes and close."""
        if not self._current_user:
            self.dismiss(False)
            return

        # Apply pending changes
        if 'name' in self._pending_changes:
            self._current_user.name = self._pending_changes['name']

        if 'show_clock' in self._pending_changes:
            self._current_user.show_clock = self._pending_changes['show_clock']

        if 'state_display_count' in self._pending_changes:
            self._current_user.state_display_count = self._pending_changes['state_display_count']

        # Add new state entry if presence or status changed
        if 'presence' in self._pending_changes or 'status' in self._pending_changes:
            self._current_user.add_state(
                iter=self._session.iteration,
                presence=self._pending_changes.get('presence'),
                status=self._pending_changes.get('status'),
            )

        self._session.save()
        self._pending_changes.clear()
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Cancel and close."""
        self.dismiss(False)
