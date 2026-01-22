# RFC: Draft Signal Channel Enhancement

**Status**: Complete (core implementation done)
**Date**: 2026-01-18
**Branch**: `draft-signal-channel`

## Summary

Enhance the draft-based dialogue system to function as a bidirectional signal channel between mind and user, not just a response buffer.

## Context

The v1.2 draft-dialogue system was implemented (commit `786c1be`). During testing, the mind identified an affordance problem: drafts accumulate but the mind has no way to manage or signal with them beyond adding more.

This RFC proposes treating the draft buffer as a **signal channel** where:
- Draft length/frequency carries meaning
- Opt-out (no draft) demands user attention
- Short ALL-CAPS words serve as signals to both future iterations AND user
- The user watches the buffer evolve in real-time

## Design Decisions (Confirmed)

### 1. Unlimited Storage, Limited Display

- **Storage**: Keep ALL drafts forever with absolute indices (1, 2, 3...)
- **Display to mind**: max(chars: 2000, count: 16) - whichever limit hit first
- **Display to user**: Full buffer visible

**Implementation status**: ✅ Done
- `DialoguePool` no longer prunes drafts
- `Draft` has `index` field (absolute, 1-based)
- `get_drafts_for_display(max_chars, max_count)` for mind
- `get_all_drafts()` for user
- Config: `draft_display_chars=2000`, `draft_display_count=16`

### 2. Absolute Indexing

Drafts have permanent indices within each exchange:
- `#1` is always first draft, `#2` second, etc.
- User accepts by absolute index: `mind accept 3`
- History records which draft was accepted: `accepted_draft_index`

**Implementation status**: ✅ Done
- CLI updated to use absolute indices
- Accept command defaults to latest draft's index

### 3. Strict Mode (iterations blocked when idle)

- Cannot run `mind step` or `mind run` when not drafting
- Must send a message first
- Prevents "free thinking" mode - all activity is in response to user

**Implementation status**: ✅ Done (from earlier)

## Design Decisions (Implemented)

### 4. Minimal Signal Vocabulary

Two signals only - one hard, one soft:

| Signal | Meaning | Cost |
|--------|---------|------|
| (no draft) | **HARD**: User attention required NOW | 0 chars |
| `+1` | **SOFT**: Latest draft is publishable, still iterating | 2 chars |

**Implementation status**: ✅ Done - added to system_prompt_v1.2.md

### 5. Opt-Out as Hard Signal

When mind outputs no `draft:`:
- Current buffer content stands
- This is a **signal spike** demanding user attention
- Mind is saying "look at what's there"
- User should check buffer and act

If the top/latest draft is correct and complete:
- Mind STFUs (stops drafting)
- User *must* pay attention
- This is integrated signal channel behavior

**Implementation status**: ✅ Done - documented in SIGNAL CHANNEL section

### 6. Buffer Dynamics

The `+1` signal creates self-limiting behavior:
- Too many `+1`s push real content off display window
- If best draft at risk of being lost: regenerate it fresh
- When best == only visible real draft: revert to hard signal (silence)

Mind learns to manage buffer visibility as a shared attention resource.

**Implementation status**: ✅ Done - documented in system prompt

### 7. User Presence Detection (Future)

When `user_seen` is being continuously set:
- User is watching the buffer in real-time
- Mind might behave differently

When user is absent:
- Mind free to cycle buffer
- When user returns, can summarize what was being thought about

## Implementation Remaining

### Immediate (this session) - COMPLETE

1. ~~Update config params~~ ✅
2. ~~DialoguePool: unlimited storage, absolute indices~~ ✅
3. ~~session_v2: new draft methods~~ ✅
4. ~~format_input: display limits~~ ✅
5. ~~CLI: absolute index accept~~ ✅
6. ~~System prompt: signal vocabulary, opt-out, buffer dynamics~~ ✅
7. ~~CLAUDE.md~~ ✅
8. ~~Draft archive (append-only JSONL)~~ ✅
9. ~~History CLI standardization~~ ✅

### Next Session

1. Review and refine signal vocabulary
2. Consider how to document opt-out behavior for mind
3. Test the signal channel in practice
4. Consider `user_seen` real-time update mechanism
5. Consider how mind learns user's name

## Files Changed (This Session)

```
Modified:
- src/core/dialogue_pool.py    # Unlimited storage, absolute indices
- src/core/session_v2.py       # New config params, draft methods
- src/core/mind_v2.py          # Display-limited format_input
- src/mind/runner.py           # Pass display-limited drafts
- scripts/mind.py              # Absolute index CLI

Not yet updated:
- docs/system_prompt_v1.2.md   # Needs signal vocabulary
- CLAUDE.md                    # Needs refresh
```

## Key Insight

The draft buffer is not just a "response queue" - it's a **communication channel** where:
- **Content** = what the mind wants to say
- **Length** = how much it has to say right now
- **Frequency** = how actively it's engaging
- **Silence** = a demand for attention
- **Short signals** = metadata about the content

This transforms the protocol from "mind produces drafts, user picks one" to "mind and user communicate through the draft buffer state."

## Open Questions

1. Should there be a way for mind to explicitly clear/reset the buffer?
2. How does the mind learn the user's name? (System prompt? First message parse?)
3. Should signal words be in a reserved namespace to avoid confusion?
4. What happens if user never accepts and buffer grows very large?
5. How to handle the transition when user returns after being away?

## Testing Notes

The mind reported the original affordance problem at iteration ~40 with 4 unaccepted drafts. This feedback directly informed the design of:
- Unlimited storage (no pruning frustration)
- Signal vocabulary (explicit communication)
- Opt-out semantics (attention demand)

---

*To continue: Load this RFC, review implementation status, proceed with system prompt and CLAUDE.md updates.*
