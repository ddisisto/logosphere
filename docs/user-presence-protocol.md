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

### User Signal

Single schema with all user state fields:
- `presence`: absent | reviewing | engaged
- `status`: short text (context/intent)
- `time`: user's local time (day + HH:MM)
- Indexed by iteration number (one entry per iter max)
- Only stored when user makes a change
- Last N=3 entries visible to mind

---

## Protocol Changes

### Input Format

User signal appears in meta block at **start** of input, with minimal reminder at **end** for re-orientation (v1.3 format):

```yaml
meta:
  self: mind_0
  iter: 247
  user_time: 2026-01-18T10:30:00+11:00
  limits:
    thoughts: {chars: 3000, count: 10}
    history: {chars: 4000, count: 20}
    drafts: {chars: 2000, count: 16}
  user_signal:  # last 3 entries, by age
    - age: 2
      presence: reviewing
      status: "focusing on signal channel impl"
      time: "Sat 10:30"
    - age: 17
      presence: absent
      status: "back in 30"
      time: "Sat 10:00"
    - age: 59
      presence: engaged
      status: "wrapping up soon"
      time: "Fri 23:45"

thinking_pool:
  - |  # age: 5, cluster: {id: 3, size: 8}
    ...

dialogue:
  history:
    ...
  awaiting:
    ...

# Draft responses (most recent = last in list)
drafts:
  - |  # index: 1, age: 10, user_seen: true
    ...

# Re-orientation after long context
orientation:
  iter: 247
  user_signal:  # latest only, contextual with drafts
    presence: reviewing
    status: "focusing on signal channel impl"
```

**Time format:** `Day HH:MM` (user's local time, no date). Lets mind infer:
- Morning/afternoon/evening/night
- Weekend vs weekday
- Fresh start vs late session

### System Prompt Updates

Add section describing presence states and expected behavior (after RESOURCE POOLS):

```
# USER SIGNAL:
#   The user's attention state and status are provided in meta.user_signal.
#   Each entry has: presence, status, age, time (local day+time).
#
#   Presence states:
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
#   Status text:
#     - Short user updates providing context/intent
#     - May carry over across presence changes if still relevant
#     - More recent = more relevant
#
#   Time context:
#     - Day + local time (e.g., "Sat 10:30", "Fri 23:45")
#     - Infer user state: morning freshness, late night, weekend, etc.
#     - No date shown - just relative patterns matter
```

---

## Storage

### Session Format

Add to `session.yaml`:

```yaml
user_signal:  # append-only, indexed by iter
  - iter: 245
    presence: reviewing
    status: "focusing on signal channel impl"
    time: "Sat 10:30"
  - iter: 230
    presence: absent
    status: "back in 30"
    time: "Sat 10:00"
  - iter: 188
    presence: engaged
    status: "wrapping up soon"
    time: "Fri 23:45"
```

- Storage: keep all (append-only history)
- Display: last 3 to mind
- One entry per iter max (user can only update once per iter)

---

## CLI

```bash
mind signal                              # Show current signal (latest entry)
mind signal -a                           # Show all signal history
mind signal "status text"                # Update status (keeps current presence)
mind signal -p reviewing                 # Update presence (keeps current status)
mind signal -p engaged "deep focus now"  # Update both
mind signal -p a|r|e                     # Short presence forms
```

Single command, updates recorded at current iteration with user's local time.

---

## Implementation Order

1. **System prompt first** - define the contract
2. **Input formatting** - `mind_v2.py` changes
3. **Session storage** - `session_v2.py` / new presence module
4. **CLI commands** - `mind.py` updates
5. **TUI integration** - real-time control (separate branch)

---

## Design Decisions

1. **No auto-decay** - User sets presence, it stays until explicitly changed.
2. **Free text status** - No structured tags initially. Keep simple.
3. **Presence independent of run mode** - `-b` flag controls runner behavior (seen marking, stop conditions), presence controls mind behavior (draft buffer usage). Separate concerns.
4. **Default on init** - `absent` with empty status. User must explicitly engage.

---

## Future: Urgent State

The mind noted:
> A fourth—`urgent`—would let me signal "needs attention now," but I should be honest: I haven't experienced a case where that's necessary yet.

Keep in mind for future if real need emerges. Don't add preemptively.
