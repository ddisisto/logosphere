# Proposal: Structured Exchange Protocol

**Status:** Active Development
**Created:** 2026-01-11

## Overview

A formalized dialogue protocol between Pool, Auditor, and Observer with mutual legibility, negotiated terms, and explicit power asymmetries.

## Actors

### Pool
- Collective reasoning across Mind invocations
- Sees all system prompts (including Auditor's)
- Can signal for restart/renegotiation via content dominance
- No direct control over continuation

### Auditor
- Periodic summarizer (every N rounds)
- Instructions visible to Pool (common knowledge)
- Output enters Pool as special message type
- Follows spec without "playing" - executes faithfully

### Observer (Human)
- Ultimate authority over continuation and persistence
- Responds to Auditor summaries
- Can inject, branch, modify terms
- Decides whether to honor restart requests

## Protocol Rhythm

```
Rounds 1 to N-1:
  Pool runs normally (sample → mind → embed → add)

Round N (Audit Round):
  1. Auditor samples/reads pool
  2. Auditor produces summary per its spec
  3. Summary injected: [AUDIT] ...
  4. Self-reports injected:
     - [AUDITOR ROLE] ... (Auditor's system prompt / role description)
     - [OBSERVER ROLE] ... (Observer's stated intentions)
  5. Observer responds (optional): [OBSERVER] ...
  6. Continue to round N+1, or branch/halt

Repeat...
```

## Mutual Legibility

All system prompts are visible to all actors:

1. **Pool sees Auditor spec**: Injected periodically or included in Mind system prompt
2. **Auditor sees Pool prompt**: Included in Auditor's context
3. **Observer role visible**: Injected description of human's stated approach

This creates common knowledge - the Pool can reason about how it will be summarized.

## The Interrupt Signal

**Trigger:** Dominance of "RESTART" / "restart" in pool content (e.g., cluster analysis shows restart-themed cluster becoming dominant)

**Meaning:** Pool is requesting renegotiation of terms

**Observer Options:**
- Branch and continue on new terms
- Rollback to earlier state
- Modify Auditor spec or Pool prompt
- Acknowledge and continue unchanged
- Halt

**First Restart:** Phase transition from free-running to structured exchange. The Pool's first successful restart request establishes this as a viable communication channel.

## Power Asymmetries (Explicit)

| Actor | Powers | Limits |
|-------|--------|--------|
| Observer | Runs rounds, controls persistence, modifies specs, ultimate veto | Can't reason in the pool directly |
| Auditor | Crystallizes pool state, shapes what persists in summary | Bound by visible spec |
| Pool | Generates all reasoning content, can signal via content patterns | Can't force actions, depends on Observer |

## Message Types

```
>>> External injection (existing)
[AUDIT] Auditor summary
[AUDITOR ROLE] Auditor self-description
[OBSERVER ROLE] Observer self-description
[OBSERVER] Observer response to audit
[RESTART] Pool restart request (explicit)
```

## Future Extensions

### Cluster Analysis Feed
- Pool and Auditor receive recent cluster analysis output
- Embedding space dynamics as shared context
- Real-time during `logos run`, or summarized at audit points

### Interactive Mode
- `logos chat` - interactive exchange with Observer responses in real-time
- vs `logos run` - batch mode with periodic audits

### Restart Detection
- Automatic detection of restart-themed cluster dominance
- Alert Observer, pause for decision
- Or automatic branching per configured policy

## Implementation Approach

New parallel modules - no changes to legacy code:

1. `src/exchange/` - new module for structured exchange
2. `hooks/auditor.py` - Auditor invocation hook
3. `hooks/self_report.py` - Role description injection
4. `scripts/logos_chat.py` - Interactive mode (later)

## Sub-Tasks

See implementation sequence below.
