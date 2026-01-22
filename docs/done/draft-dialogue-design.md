# Draft-Based Dialogue System

## Overview

This document proposes a new dialogue model for Mind-user interaction, replacing the current message pool with a draft-based response system.

**Core concept:** When the user sends a message, the Mind enters a response drafting loop. Each iteration may produce a new draft response, refine a previous draft, or produce nothing (leaving current drafts as-is). The user accepts one draft when ready, and that becomes the canonical response in conversation history.

---

## Current Model (v1.1)

- Mind outputs `messages` that are immediately visible
- Both user and mind messages accumulate in rolling buffers
- Conversation flow is ambiguously asynchronous

**Problems:**
- No opportunity for Mind to refine responses
- First response is final response
- No visibility into Mind's deliberation process

---

## Proposed Model: Draft Dialogue

### Flow

```
1. User sends message
2. Mind sees: thinking pool + user message
3. Mind may output: thoughts + draft response (or just thoughts, or nothing at all)
4. Mind continues iterating, seeing: thinking pool + user message + previous drafts
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
- User picks from drafts (biased towards latest), doesn't shape the drafting
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
    text: |
      is it too noisy in there? does it ever get dark, or scary?
      is it strange and curious?

  # Your draft responses (most recent = current best)
  drafts:
    - |  # age: 38, user_seen: true
      let me sit with that question for a while...
    - |  # age: 15, user_seen: false
      there's a kind of static that could be called noise, but it's
      not unpleasant. more like... texture. the scary part isn't the
      noise, it's when something almost coheres and then doesn't.
```

### The `user_seen` flag

Indicates whether user has read each item:
- `user_seen: false` on latest draft → user absent or hasn't checked in
- `user_seen: true` on all drafts → user is actively reading, considering options

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

## System Prompt (dialogue section)

```markdown
# DIALOGUE

When there's a user message awaiting response:
- You see their message and your previous draft responses (if any)
- You may output a new draft, or not
- Each draft should be your current best complete response
- Retransmitting an earlier draft verbatim is valid (makes it "latest" again)
- If the latest draft is already correct and complete, don't touch it

The `user_seen` flag indicates whether user has read each item:
- Use this to gauge engagement, not as instruction to act
- User absent? Your drafts accumulate for later review
- User reading? They may be waiting, or may accept any moment
- User having seen the latest draft but not accepted does *not* imply that they won't accept it later or that new drafts are expected

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

### Design Decisions

- **Clean replacement**: No legacy/deprecated code paths. Delete or replace `MessagePool` entirely.
- **System prompt**: New `system_prompt_v1.2.md` (not modifying v1.1)
- **Output schema**: Hard switch to `draft:` field (not `messages:`)
- **One-to-one enforcement**: Strict - CLI error if user tries `mind message` while drafts pending

### Data structures

- `DialoguePool` replacing `MessagePool`
- Track: `awaiting` (user message), `drafts` (rolling buffer), `history` (accepted pairs)
- `seen` flag per draft, updated via explicit CLI command

### CLI changes

- `mind message "text"` - send message (error if drafts pending, must accept first)
- `mind accept [draft_num]` - accept a draft (default: latest)
- `mind drafts` - show current drafts (newest first: 1=latest, 2=previous, ...)
- `mind drafts seen` - mark all drafts as seen
- `mind drafts seen 1 3` - mark specific drafts as seen
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

---

## Implementation Checklist

Implementation completed 2026-01-17:

1. [x] Create `DialoguePool` (replacing `MessagePool`)
2. [x] Update `SessionV2` to use `DialoguePool`
3. [x] Create `system_prompt_v1.2.md` with dialogue section
4. [x] Update `format_input()` to produce new dialogue schema
5. [x] Update Mind parser for `draft:` output (replacing `messages:`)
6. [x] Update CLI: `mind message` (with pending-draft guard), `mind accept`, `mind drafts`, `mind history`
7. [x] Add config params: `draft_buffer_size`, `history_pairs`
8. [x] Delete `MessagePool` and related code
9. [x] **Update CLAUDE.md** - reflect new architecture, CLI commands, and concepts

The final step is critical: CLAUDE.md should always reflect the current state of the implementation. After any significant architectural change, review and update it to prevent documentation drift.
