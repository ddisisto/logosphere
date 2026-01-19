# TUI Design

## Overview

Textual-based TUI for Logosphere, providing real-time visibility into draft buffer, iteration progress, and signal state. Designed for modularity and code reuse with CLI.

## Layout

```
┌─────────────────────────────────────┬──────────────────────┐
│                                     │ StatusPanel          │
│  MainView                           │  - presence          │
│   ├─ DraftBufferView (default)      │  - signal_state      │
│   └─ HistoryView (alternate)        │  - session meta      │
│                                     ├──────────────────────┤
│  [scrollable]                       │ IterationLog         │
│                                     │  - per-iter summary  │
│                                     │  [scrollable]        │
├─────────────────────────────────────┴──────────────────────┤
│ [Run] [Pause]                              iter: 247       │
└────────────────────────────────────────────────────────────┘
```

## Components

### MainView
- **DraftBufferView** (default): Scrolling list of drafts, newest at bottom
  - Each draft shows: index, age, text, seen status
  - Visual distinction for short signals (<=16 chars) vs long drafts
- **HistoryView** (alternate): Conversation history
  - Toggle via keybinding (e.g., `h`)

### StatusPanel (sidebar top)
- Current presence state (absent/reviewing/engaged)
- Signal state: `consecutive_hard: N / threshold: M`
- Session metadata: iteration count, pool sizes
- Awaiting message preview (if drafting)

### IterationLog (sidebar bottom)
- Per-iteration summaries, scrollable
- Each entry shows:
  - Iteration number
  - Thoughts: count + total chars
  - Draft: flag + length (or signal indicator if none)
  - Presence transition (if changed that iteration)

### Controls
- **Run**: Start iteration loop (conditional - only when drafting)
- **Pause**: Stop iteration loop
- Keybindings for common ops (accept, message, presence change)

## Architecture

### Code Reuse Strategy

Refactor to separate logic from presentation:

```
src/
├── core/           # unchanged - data model
├── mind/
│   ├── runner.py   # emits events via EventEmitter
│   ├── ops.py      # shared operations (send_message, accept_draft, etc.)
│   ├── events.py   # EventEmitter + event dataclasses
│   └── config.py
├── tui/            # Phase 2+
│   ├── app.py      # Textual app
│   ├── views/      # DraftBufferView, HistoryView
│   └── panels/     # StatusPanel, IterationLog
scripts/
└── mind.py         # CLI - uses ops.py
```

### Event-Based Runner

Runner emits structured events via `EventEmitter` (see `src/mind/events.py`):

```python
# Key event types (EventType enum)
ITERATION_START, ITERATION_COMPLETE, RUN_START, RUN_STOP, SIGNAL_DETECTED

# Example: subscribe to iteration events
runner.events.on(EventType.ITERATION_COMPLETE, my_handler)
```

Event dataclasses include: `IterationStartEvent`, `IterationCompleteEvent`, `RunStartEvent`, `RunStopEvent`, `SignalDetectedEvent`.

Subscribers (CLI verbose mode, TUI) handle rendering.

### Shared Operations (ops.py)

Extract from CLI commands:

```python
def send_message(session: SessionV2, text: str) -> None
def accept_draft(session: SessionV2, index: Optional[int] = None) -> Draft
def mark_drafts_seen(session: SessionV2, indices: Optional[list[int]] = None) -> int
def set_presence(session: SessionV2, presence: PresenceState, status: Optional[str] = None) -> UserSignal
def get_session_status(session: SessionV2) -> dict
```

CLI and TUI both call these, handle output differently.

## Implementation Phases

### Phase 1: Refactor for Reuse ✓ COMPLETE
1. ✓ Create `src/mind/ops.py` with shared operations + result types
2. ✓ Refactor CLI to use ops.py (removed duplicate helpers)
3. ✓ Create `src/mind/events.py` with EventEmitter + event dataclasses
4. ✓ Runner emits events alongside verbose prints

### Phase 2: Basic TUI Shell ✓ COMPLETE
1. ✓ Textual app with layout structure (3:1 main/sidebar)
2. ✓ Real data in all views (DraftBufferView, HistoryView, StatusPanel)
3. ✓ Session loading from `~/.mind_session`
4. ✓ View toggle (`h` key) between drafts and history
5. ✓ Auto-set presence to "reviewing" on launch
6. ✓ Entry point: `scripts/tui.py`

### Phase 3: Live Views
1. Views update when session changes (reactive)
2. StatusPanel signal_state from runner (currently placeholder)
3. Refresh on external session changes

### Phase 4: Iteration Control
1. Run/Pause controls
2. IterationLog with live updates
3. Event subscription from runner

### Phase 5: Full Interaction
1. Message input
2. Draft acceptance
3. Presence controls
4. Keybindings

## Keybindings (Draft)

| Key | Action |
|-----|--------|
| `r` | Run iterations |
| `p` | Pause |
| `a` | Accept latest draft |
| `1-9` | Accept draft N |
| `h` | Toggle history view |
| `m` | Message input |
| `P` | Cycle presence |
| `q` | Quit |

## Dependencies

- `textual` - TUI framework
- Existing: session_v2, dialogue_pool, runner

## Design Decisions

Resolved during Phase 2:

- **Draft buffer display**: Same config limits as `format_input` (user sees what model sees)
- **IterationLog retention**: 3 entries, latest at top, lazy-load scroll later
- **Message input**: Modal, required after accept/before run, unavailable otherwise
- **Auto-presence**: Yes, set to "reviewing" on TUI launch
- **Session selection**: Follow CLI pattern (`~/.mind_session`)
- **Threading**: Textual `@work` decorator for runner (Phase 4)
- **Error display**: Blocking modal (Phase 5)
