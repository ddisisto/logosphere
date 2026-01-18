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
│   ├── runner.py   # emit events instead of prints
│   ├── ops.py      # NEW: shared operations
│   └── config.py
├── tui/
│   ├── app.py      # Textual app
│   ├── views/      # DraftBufferView, HistoryView
│   ├── panels/     # StatusPanel, IterationLog
│   └── events.py   # TUI event handling
scripts/
└── mind.py         # CLI - thin wrapper calling ops.py
```

### Event-Based Runner

Runner emits structured events rather than printing:

```python
@dataclass
class IterationEvent:
    iteration: int
    thoughts_added: int
    thoughts_chars: int
    draft_added: bool
    draft_length: Optional[int]
    hard_signal: bool
    signal_state: dict  # {consecutive_hard, threshold}

@dataclass
class StopEvent:
    reason: str  # "draft", "hard_signal", "true_silence", "max_reached"
    iterations_run: int
```

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

### Phase 1: Refactor for Reuse
1. Create `ops.py` with shared operations
2. Refactor CLI to use ops.py
3. Add event emission to runner (keep prints for now, emit alongside)

### Phase 2: Basic TUI Shell
1. Textual app with layout structure
2. Static panels (no live data)
3. Session loading

### Phase 3: Live Views
1. DraftBufferView with real data
2. StatusPanel with session state
3. HistoryView toggle

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

## Open Questions

- Draft buffer: show all or limit display like mind sees?
- IterationLog: how many iterations to retain?
- Message input: inline or modal?
- Should TUI auto-set presence to "reviewing" on launch?
