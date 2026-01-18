"""
Event types for Mind runner.

These events allow subscribers (CLI, TUI) to receive structured
information about iteration progress without coupling to print statements.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import Enum


class EventType(Enum):
    """Types of events emitted by the runner."""
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"
    THOUGHT_ADDED = "thought_added"
    DRAFT_ADDED = "draft_added"
    SIGNAL_DETECTED = "signal_detected"
    RUN_START = "run_start"
    RUN_STOP = "run_stop"


@dataclass
class IterationStartEvent:
    """Emitted when an iteration begins."""
    iteration: int
    thoughts_sampled: int
    state: str  # "drafting" or "idle"


@dataclass
class IterationCompleteEvent:
    """Emitted when an iteration completes."""
    iteration: int
    thoughts_added: int
    thoughts_chars: int
    draft_added: bool
    draft_length: Optional[int]
    draft_text: Optional[str]
    hard_signal: bool
    signal_state: dict  # {consecutive_hard, threshold}
    skipped: bool


@dataclass
class ThoughtAddedEvent:
    """Emitted when a thought is added to the pool."""
    iteration: int
    text: str
    vector_id: int


@dataclass
class DraftAddedEvent:
    """Emitted when a draft is added."""
    iteration: int
    index: int
    text: str
    is_signal: bool  # True if <=16 chars (soft signal)


@dataclass
class SignalDetectedEvent:
    """Emitted when a signal is detected."""
    iteration: int
    signal_type: str  # "hard" or "soft"
    consecutive_hard: int
    threshold: int
    message: str


@dataclass
class RunStartEvent:
    """Emitted when a run begins."""
    mode: str  # "observe" or "background"
    iterations: Optional[int]  # None = run until stop
    max_iterations: int


@dataclass
class RunStopEvent:
    """Emitted when a run stops."""
    reason: str  # "draft", "hard_signal", "true_silence", "max_reached", "exact_count"
    iterations_run: int
    total_thoughts: int
    total_drafts: int


# Type alias for event callbacks
EventCallback = Callable[[Any], None]


class EventEmitter:
    """
    Simple event emitter for runner events.

    Usage:
        emitter = EventEmitter()
        emitter.on(EventType.ITERATION_COMPLETE, my_handler)
        emitter.emit(EventType.ITERATION_COMPLETE, event)
    """

    def __init__(self):
        self._handlers: dict[EventType, list[EventCallback]] = {}

    def on(self, event_type: EventType, handler: EventCallback) -> None:
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def off(self, event_type: EventType, handler: EventCallback) -> None:
        """Unregister a handler."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    def emit(self, event_type: EventType, event: Any) -> None:
        """Emit an event to all registered handlers."""
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                handler(event)

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
