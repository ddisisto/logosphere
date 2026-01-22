# Dialogue v2 Design

Replaces `dialogue_pool.py` with a cleaner append-only model.

## Goals

1. **Simpler state model**: No separate "awaiting" - state derived from last message
2. **Append-only storage**: Two files, both append-only, easy to reason about
3. **Clear indexing**: Draft index is 0-based within message round, reset on each user message
4. **Separated concerns**: Messages and drafts in separate files

## Storage

```
session/
├── messages.yaml    # Append-only: user messages + accepted drafts
└── drafts.yaml      # Append-only: all drafts, grouped by user message
```

### messages.yaml

Append-only list of messages. Each entry is either a user message or an accepted mind response.

```yaml
- role: user
  iter: 5
  time: "2026-01-20T10:00:00+00:00"
  text: |
    Hello, how are you?

- role: mind
  iter: 12
  time: "2026-01-20T10:05:00+00:00"
  text: |
    I'm doing well, thank you for asking.
  draft_index: 2          # Which draft was accepted (0-indexed)
  user_message_iter: 5    # Links back to the user message

- role: user
  iter: 15
  time: "2026-01-20T10:10:00+00:00"
  text: |
    Great to hear!
```

**State derivation:**
- If last message is `role: user` → DRAFTING state (awaiting response)
- If last message is `role: mind` or empty → IDLE state
- No separate "awaiting" field needed

### drafts.yaml

Append-only list of all drafts. Grouped logically by user message.

```yaml
- user_message_iter: 5    # Which user message this responds to
  index: 0                # 0-indexed within this message round
  iter: 7                 # Global iteration when created
  time: "2026-01-20T10:01:00+00:00"
  text: |
    First draft attempt...
  seen: false

- user_message_iter: 5
  index: 1
  iter: 9
  time: "2026-01-20T10:02:00+00:00"
  text: "+1"
  seen: true

- user_message_iter: 5
  index: 2
  iter: 12
  time: "2026-01-20T10:05:00+00:00"
  text: |
    I'm doing well, thank you for asking.
  seen: true
```

**Index assignment:**
- When a new user message is sent, draft index resets to 0
- Each new draft for that message gets the next index (0, 1, 2...)
- Index is the primary reference for accept/display operations

**Querying drafts for current message:**
- Filter where `user_message_iter == last_user_message.iter`

## Data Classes

### Message

```python
@dataclass
class Message:
    role: Literal["user", "mind"]
    iter: int
    time: str
    text: str
    # Only for mind messages:
    draft_index: int | None = None        # Which draft was accepted
    user_message_iter: int | None = None  # Links to user message
```

### Draft

```python
@dataclass
class Draft:
    user_message_iter: int  # Which user message this responds to
    index: int              # 0-indexed within message round
    iter: int               # Global iteration when created
    time: str
    text: str
    seen: bool = False
```

## Module Structure

```
src/core/
├── message_store.py    # MessageStore class
├── draft_store.py      # DraftStore class
└── dialogue_pool.py    # OLD - to be removed after migration
```

### MessageStore (`message_store.py`)

```python
class MessageStore:
    """Append-only message storage."""

    def __init__(self, path: Path): ...

    # Properties
    @property
    def is_drafting(self) -> bool:
        """True if last message is from user."""

    @property
    def last_user_message(self) -> Message | None:
        """Get the last user message, if in drafting state."""

    # Operations
    def add_user_message(self, text: str, iter: int) -> Message:
        """Add a user message. Raises if already drafting."""

    def add_mind_message(
        self,
        text: str,
        iter: int,
        draft_index: int,
        user_message_iter: int,
    ) -> Message:
        """Add accepted mind response. Raises if not drafting."""

    # Queries
    def get_all(self) -> list[Message]:
        """Get all messages (oldest first)."""

    def get_history_for_display(
        self,
        max_count: int,
        max_chars: int,
    ) -> list[Message]:
        """Get recent history within limits."""

    # Persistence
    def save(self) -> None: ...
    def _load(self) -> None: ...
```

### DraftStore (`draft_store.py`)

```python
class DraftStore:
    """Append-only draft storage."""

    def __init__(self, path: Path): ...

    # Operations
    def add_draft(
        self,
        user_message_iter: int,
        text: str,
        iter: int,
        seen: bool = False,
    ) -> Draft:
        """Add a draft. Returns the draft with assigned index."""

    def mark_seen(
        self,
        user_message_iter: int,
        indices: list[int] | None = None,
    ) -> int:
        """Mark drafts as seen. None = all. Returns count marked."""

    # Queries
    def get_drafts_for_message(self, user_message_iter: int) -> list[Draft]:
        """Get all drafts for a user message (oldest first)."""

    def get_draft(self, user_message_iter: int, index: int) -> Draft | None:
        """Get specific draft by index."""

    def get_latest_draft(self, user_message_iter: int) -> Draft | None:
        """Get the most recent draft for a message."""

    def get_drafts_for_display(
        self,
        user_message_iter: int,
        max_count: int,
        max_chars: int,
    ) -> list[Draft]:
        """Get recent drafts within limits (newest first)."""

    def next_index(self, user_message_iter: int) -> int:
        """Get the next draft index for a message (for internal use)."""

    # Persistence
    def save(self) -> None: ...
    def _load(self) -> None: ...
```

## Session Integration

SessionV2 will use both stores:

```python
class SessionV2:
    def __init__(self, session_dir: Path):
        # ...
        self.messages = MessageStore(session_dir / "messages.yaml")
        self.drafts = DraftStore(session_dir / "drafts.yaml")

    @property
    def is_drafting(self) -> bool:
        return self.messages.is_drafting

    def send_message(self, text: str) -> Message:
        """Send user message."""
        return self.messages.add_user_message(text, self.iteration)

    def add_draft(self, text: str, seen: bool = False) -> Draft:
        """Add draft for current message."""
        user_msg = self.messages.last_user_message
        if user_msg is None:
            raise RuntimeError("Not in drafting state")
        return self.drafts.add_draft(
            user_message_iter=user_msg.iter,
            text=text,
            iter=self.iteration,
            seen=seen,
        )

    def accept_draft(self, index: int) -> Draft:
        """Accept a draft by index."""
        user_msg = self.messages.last_user_message
        if user_msg is None:
            raise RuntimeError("Not in drafting state")

        draft = self.drafts.get_draft(user_msg.iter, index)
        if draft is None:
            raise IndexError(f"Draft {index} not found")

        self.messages.add_mind_message(
            text=draft.text,
            iter=draft.iter,
            draft_index=index,
            user_message_iter=user_msg.iter,
        )
        return draft

    def get_current_drafts(self) -> list[Draft]:
        """Get drafts for current message round."""
        user_msg = self.messages.last_user_message
        if user_msg is None:
            return []
        return self.drafts.get_drafts_for_message(user_msg.iter)
```

## Operations Changes

`ops.py` simplifies significantly:

```python
def accept_draft(session: SessionV2, index: int | None = None) -> AcceptDraftResult:
    """Accept draft by index. None = latest (highest index)."""
    if not session.is_drafting:
        return AcceptDraftResult(success=False, error="Not in drafting state")

    drafts = session.get_current_drafts()
    if not drafts:
        return AcceptDraftResult(success=False, error="No drafts available")

    if index is None:
        index = drafts[-1].index  # Latest

    try:
        draft = session.accept_draft(index)
        session.save()
        return AcceptDraftResult(success=True, draft=draft)
    except IndexError as e:
        valid = [d.index for d in drafts]
        return AcceptDraftResult(
            success=False,
            error=f"Draft {index} not found. Valid: {valid}"
        )
```

No more `resolve_draft_ref` with iter-vs-offset confusion.

## CLI Changes

```bash
mind drafts           # Shows drafts 0, 1, 2... for current message
mind accept           # Accept latest (highest index)
mind accept 2         # Accept draft index 2
mind message "text"   # Send message (only if idle)
```

Negative offsets removed - just use the displayed index.

## TUI Changes

- DraftBufferView shows drafts with `#0`, `#1`, `#2`...
- Selection + `a` key accepts by index
- `get_selected_draft().index` passed directly to `ops.accept_draft()`

## Migration

Not handled in code. For existing sessions:
1. External script can convert `dialogue/pool.yaml` → `messages.yaml` + `drafts.yaml`
2. Or simply start fresh sessions with new format

## Implementation Order

1. Create `src/core/message_store.py`
2. Create `src/core/draft_store.py`
3. Update `SessionV2` to use new stores (keep old dialogue_pool temporarily)
4. Update `ops.py` to use new model
5. Update `runner.py` draft emission
6. Update CLI commands
7. Update TUI views
8. Remove `dialogue_pool.py`
9. Update CLAUDE.md

## Open Questions

None currently - design is straightforward.
