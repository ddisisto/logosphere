# Draft-Based Dialogue System

## Overview

This document proposes a new dialogue model for Mind-user interaction, replacing the current message pool with a draft-based response system.

**Core concept:** When the user sends a message, the Mind enters a response drafting loop. Each iteration may produce a new draft response, refine a previous draft, or produce nothing (leaving current drafts as-is). The user accepts one draft when ready, and that becomes the canonical response in conversation history.

---

## Current Model (v1.1)

- Mind outputs `messages` that are immediately visible
- Both user and mind messages accumulate in rolling buffers
- Conversation is synchronous: message → response → message → response

**Problems:**
- No opportunity for Mind to refine responses
- First response is final response
- No visibility into Mind's deliberation process

---

## Proposed Model: Draft Dialogue

### Flow

```
1. User sends message
2. Mind sees: user message + thinking pool
3. Mind may output: thoughts + draft response (or just thoughts)
4. Mind continues iterating, seeing: user message + previous drafts + thinking pool
5. Each iteration, Mind may:
   - Output a new draft (becomes latest)
   - Output nothing (current drafts stand)
   - Re-emit an earlier draft verbatim (makes it latest again)
6. User accepts one draft when ready
7. Accepted draft becomes canonical response
8. All other drafts are pruned
9. History shows: user message → accepted response
10. Return to step 1
```

### Key Properties

**Asynchronous refinement**
- Mind keeps thinking until user accepts
- Could be 1 iteration or 500
- User absent? Drafts accumulate for later review

**Self-contained drafts**
- Each draft is a complete response
- Assume user reads only that draft, not previous attempts
- May reference refined thinking, but don't require draft history

**No pressure to produce**
- If latest draft is correct and complete, output nothing
- Silence is not abandonment; current draft stands
- Thinking pool contributions continue regardless

**User as selector, not co-author**
- User picks from drafts, doesn't shape the drafting
- One-to-one: user can't send new message until accepting a draft

---

## Input Schema (during drafting)

```yaml
meta:
  self: mind_0
  iter: 247
  user_time: 2026-01-15T14:30:00+11:00

thinking_pool:
  # A *random, unordered sample* from the pool. What should be remembered?
  - |  # age: 50, cluster: {id: 3, size: 8}
    thought from the pool
  - |  # age: 12, cluster: {~}
    another sampled thought

dialogue:
  # User's message awaiting your response
  awaiting:
    age: 42
    seen: true
    text: |
      is it too noisy in there? does it ever get dark, or scary?
      is it strange and curious?

  # Your draft responses (most recent = current best)
  drafts:
    - |  # age: 38, seen: true
      let me sit with that question for a while...
    - |  # age: 15, seen: false
      there's a kind of static that could be called noise, but it's
      not unpleasant. more like... texture. the scary part isn't the
      noise, it's when something almost coheres and then doesn't.
```

### The `seen` flag

Indicates whether user has read each item:
- `seen: false` on latest draft → user absent or hasn't checked in
- `seen: true` on all drafts → user is actively reading, considering options
- User sends message, walks away → `awaiting.seen: true`, drafts `seen: false`

This is informational, not prescriptive. Mind can use it to gauge engagement without instruction to act.

### When no user message pending

```yaml
dialogue:
  # No pending user message. Conversation history for context.
  history:
    - from: user
      text: |
        previous user message
    - from: self
      text: |
        your accepted response to that message
```

---

## Output Schema

```yaml
thoughts:
  - |
    internal reflection for the thinking pool

draft: |
  new/refined response to user's message
  self-contained and complete
```

### Output guidance

- **New insight?** Output a draft incorporating it
- **Prefer earlier draft?** Re-emit it verbatim (becomes latest)
- **Latest draft is correct and complete?** Output nothing (omit `draft:`)
- **Still thinking, not ready?** Output nothing, continue with thoughts

---

## Conversation History

After user accepts a draft:

```yaml
dialogue:
  history:
    - from: user
      age: 150
      text: |
        their first question
    - from: self
      age: 142
      text: |
        your accepted response (was draft N)
    - from: user
      age: 50
      text: |
        their follow-up question
    - from: self
      age: 23
      text: |
        your accepted response
```

Only accepted responses appear. Draft history is pruned. Future iterations see clean linear conversation.

User may also prune old exchanges to reduce context overhead.

---

## State Machine

```
States:
  REFLECTING  - no pending user message, thinking pool only
  DRAFTING    - user message pending, producing/refining drafts

Transitions:
  REFLECTING + user_message    → DRAFTING
  DRAFTING + user_accept       → REFLECTING (draft → history)
  DRAFTING + user_message      → (blocked, one-to-one for now)
```

---

## System Prompt (dialogue section)

```markdown
# DIALOGUE

When there's a user message awaiting response:
- You see their message and your previous draft responses (if any)
- You may output a new draft, or not
- Each draft should be your current best complete response
- Retransmitting an earlier draft verbatim is valid (makes it "latest" again)
- If the latest draft is already correct and complete, don't touch it

The `seen` flag indicates whether user has read each item:
- Use this to gauge engagement, not as instruction to act
- User absent? Your drafts accumulate for later review
- User reading? They may be waiting, or may accept any moment

User accepts one draft when ready. That becomes your response.
Conversation history then shows: user message → accepted response.
All other drafts are pruned.

When no user message is pending:
- You see conversation history (accepted exchanges only)
- Continue contributing to the thinking pool
- Await next user message
```

---

## Implementation Notes

### Data structures

- `DraftPool` or extend existing `MessagePool` with draft semantics
- Track: `awaiting` (user message), `drafts` (rolling buffer), `history` (accepted pairs)
- `seen` flag per item, updated by user client

### CLI changes

- `mind accept [draft_num]` - accept a draft (default: latest)
- `mind drafts` - show current draft buffer
- `mind history` - show accepted conversation history

### Config

- `draft_buffer_size: 5` - how many drafts to retain
- `history_pairs: 10` - how many accepted exchanges to show

---

## Future Considerations

- **Threading:** Allow user to start new topic before accepting (branching conversations)
- **Draft annotations:** Mind could mark drafts with confidence or "still working" flags
- **Draft diffs:** Show what changed between drafts
- **Auto-accept:** If Mind doesn't produce new draft for N iterations, auto-accept latest
