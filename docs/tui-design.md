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

### Session Locking

Prevents concurrent access from TUI and CLI.

**Location**: `{session_dir}/.lock`

**Contents**:
```yaml
pid: 12345
holder: "tui"  # or "cli:run", "cli:step"
started: "2026-01-19T14:30:00+00:00"
```

**Behavior**:
- **Acquired on**: TUI launch, `mind run`, `mind step`
- **Released on**: TUI exit, run/step completion
- **Read-only bypass**: `mind status` works without lock (read-only)
- **Stale detection**: Check if PID alive, auto-clear if dead
- **Recovery**: Manual removal of `.lock` file (no --force flag)

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

### Phase 3: Single-Step + Live Updates
1. Session locking (acquire on TUI launch / CLI run, release on exit)
2. Single-step iteration (`s` key) - no auto-run on startup
3. Background worker for step (Textual `@work`)
4. Event subscription → live IterationLog updates
5. Views refresh after each iteration (DraftBufferView, StatusPanel)

### Phase 4: Full Interaction
1. Message input modal (required after accept, before run)
2. Draft acceptance (`a` for latest, `1-9` for specific)
3. Presence controls (`P` to cycle)
4. Error display (blocking modal)

### Phase 5: Continuous Running
1. Run/Pause controls (`r`/`p` keys)
2. Background loop with stop conditions (draft, hard signal, true silence)
3. Observe mode toggle (mark drafts seen vs background)

## Keybindings (Draft)

| Key | Action | Phase |
|-----|--------|-------|
| `s` | Step (single iteration) | 3 |
| `h` | Toggle history view | 2 |
| `q` | Quit | 2 |
| `a` | Accept latest draft | 4 |
| `1-9` | Accept draft N | 4 |
| `m` | Message input | 4 |
| `P` | Cycle presence | 4 |
| `r` | Run iterations | 5 |
| `p` | Pause | 5 |

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
- **Threading**: Textual `@work` decorator for runner (Phase 3)
- **Error display**: Blocking modal (Phase 4)

Resolved during Phase 3 planning:

- **Session locking**: Full lock (not just iterations), exception for read-only `status`
- **Lock recovery**: Manual `.lock` file removal, no --force flag
- **Phase reorder**: Iteration control before full interaction (live updates need runner)
