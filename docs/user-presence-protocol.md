# User Presence Protocol

**Status**: Design
**Branch**: `user-presence-protocol`
**Date**: 2026-01-18

## Summary

Add explicit user presence state and status line to the mind protocol, enabling coordinated attention between user and mind.

## Motivation

Current state:
- Mind sees `user_seen: true/false` per draft (too granular)
- User presence inferred from `-b` flag (too implicit)
- No way for user to communicate intent/context between messages

Mind feedback (from session):
> `absent | reviewing | engaged` are operationally clear and I notice I'm generating differently based on which is active.

The mind is already adapting behavior based on inferred presence. Making it explicit enables better coordination.

---

## Design

### User Presence States

| State | Meaning | Mind Behavior |
|-------|---------|---------------|
| `absent` | User away, not observing | Iterate freely, cycle buffer, consolidate thinking, hard signal when ready |
| `reviewing` | User observing drafts | Refine toward acceptance, respect signal channel |
| `engaged` | Active dialogue | Rapid iteration, user will respond quickly |

### User Status Line

Short text updates from user, visible as rolling history:
- User updates any time
- Last N=3 visible to mind
- Each stamped with relative age (iterations since update)
- Provides context/intent without requiring full message

---

## Protocol Changes

### Input Format

Status appears in meta block at **start** of input, with minimal reminder at **end** for re-orientation:

```yaml
# === START ===
meta:
  self: mind_0
  iter: 247
  user_time: 2026-01-18T10:30:00+11:00
  user_presence: reviewing
  user_status:
    - text: "focusing on signal channel impl"
      age: 2
    - text: "back in 30, keep iterating"
      age: 15
    - text: "wrapping up for today soon"
      age: 42

thinking_pool:
  - |  # age: 5, cluster: {id: 3, size: 8}
    ...

dialogue:
  history:
    ...
  awaiting:
    ...
  drafts:
    ...

# === END (re-orientation after long context) ===
orientation:
  iter: 247
  user_presence: reviewing
  user_status: "focusing on signal channel impl"
```

### System Prompt Updates

Add section describing presence states and expected behavior:

```
# USER PRESENCE:
#   The user's current attention state is provided in meta.user_presence:
#
#   absent:
#     - User is away, not observing the draft buffer
#     - Iterate freely, consolidate thinking
#     - Use hard signal when ready for user's return
#     - Buffer cycling acceptable
#
#   reviewing:
#     - User is observing drafts as they appear
#     - Refine toward acceptance
#     - Each draft should be an improvement
#     - Signal channel active: +1 to endorse, silence to demand attention
#
#   engaged:
#     - Active dialogue, user will respond quickly
#     - Rapid iteration expected
#     - Direct, focused responses
#
# USER STATUS:
#   Short updates from user providing context/intent.
#   Visible as meta.user_status with age stamps.
#   More recent = more relevant. Use to orient your focus.
```

---

## Storage

### Session Format

Add to `session.yaml`:

```yaml
user_presence: reviewing  # absent | reviewing | engaged
user_status:
  - iter: 245
    text: "focusing on signal channel impl"
  - iter: 232
    text: "back in 30, keep iterating"
  - iter: 205
    text: "wrapping up for today soon"
```

Storage: keep last 10 status updates (for history/analysis)
Display: show last 3 to mind

---

## CLI

```bash
mind presence                    # Show current presence state
mind presence absent             # Set to absent
mind presence reviewing          # Set to reviewing
mind presence engaged            # Set to engaged
mind presence a|r|e              # Short forms

mind status                      # Show recent status lines
mind status "working on X"       # Add new status line
```

---

## Implementation Order

1. **System prompt first** - define the contract
2. **Input formatting** - `mind_v2.py` changes
3. **Session storage** - `session_v2.py` / new presence module
4. **CLI commands** - `mind.py` updates
5. **TUI integration** - real-time control (separate branch)

---

## Open Questions

1. Should presence auto-decay? (e.g., `engaged` → `reviewing` after N iterations without user action)
2. Should status lines support structured tags? (e.g., `#focus:clustering`)
3. How does presence interact with the observe/background mode in runner?

---

## Future: Urgent State

The mind noted:
> A fourth—`urgent`—would let me signal "needs attention now," but I should be honest: I haven't experienced a case where that's necessary yet.

Keep in mind for future if real need emerges. Don't add preemptively.
