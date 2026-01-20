"""Message input modal."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static, TextArea

from ...core.session_v2 import SessionV2


class MessageModal(ModalScreen[str | None]):
    """
    Modal for composing a message to the mind.

    Shows recent history for context, with TextArea for composing.
    Returns the message text on send, None on cancel.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "send", "Send"),
    ]

    CSS = """
    MessageModal {
        align: center middle;
    }

    #modal-container {
        width: 90%;
        max-width: 100;
        height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    #modal-title {
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }

    #history-scroll {
        height: 1fr;
        border: solid $secondary;
        margin-bottom: 1;
    }

    .history-entry {
        padding: 0 1;
        margin-bottom: 1;
    }

    .history-user {
        border-left: thick $primary;
    }

    .history-mind {
        border-left: thick $success;
    }

    .history-empty {
        padding: 1;
        color: $text-muted;
    }

    #input-section {
        height: 12;
    }

    #input-label {
        height: 2;
    }

    #message-input {
        height: 1fr;
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
        # Use same selection as mind sees (config-based limits)
        self._history = session.get_history_for_mind()
        self._current_iter = session.iteration

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label("Send Message", id="modal-title")

            # History section (scrollable, same content mind sees)
            with VerticalScroll(id="history-scroll"):
                if not self._history:
                    yield Static("[dim]No conversation history[/]", classes="history-empty")
                else:
                    for msg in self._history:
                        yield self._make_history_entry(msg)

            # Input section
            with Vertical(id="input-section"):
                yield Label("Your message:", id="input-label")
                yield TextArea(id="message-input")

            yield Static("[dim]Ctrl+S to send · Esc to cancel[/]", id="hint")

            # Buttons
            with Horizontal(id="button-row"):
                yield Button("Send", id="send-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn", variant="default")

    def on_mount(self) -> None:
        """Focus text area and scroll history to bottom."""
        self.query_one("#message-input", TextArea).focus()
        # Scroll history to bottom (newest)
        history_scroll = self.query_one("#history-scroll", VerticalScroll)
        history_scroll.scroll_end(animate=False)

    def _make_history_entry(self, msg) -> Static:
        """Create a widget for a single history entry."""
        age = self._current_iter - msg.iter
        role_label = "[bold blue]user[/]" if msg.role == "user" else "[bold green]mind[/]"
        classes = f"history-entry history-{msg.role}"

        # Format entry
        lines = [f"{role_label} · age {age}"]
        text = msg.text.strip()
        for line in text.split("\n"):
            lines.append(f"  {line}")

        return Static("\n".join(lines), classes=classes)

    def action_send(self) -> None:
        """Send the message."""
        text_area = self.query_one("#message-input", TextArea)
        text = text_area.text.strip()

        if not text:
            self.notify("Message cannot be empty", severity="warning")
            return

        self.dismiss(text)

    def action_cancel(self) -> None:
        """Cancel and close modal."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "send-btn":
            self.action_send()
        elif event.button.id == "cancel-btn":
            self.action_cancel()
