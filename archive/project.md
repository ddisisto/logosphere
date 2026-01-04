# LOGOSPHERE: Project Brief v2

## Vision

A minimal artificial life system where LLM instances form a population of stateless "Minds" exchanging messages through a shared pool. The goal is to observe memetic dynamics—drift, selection, competition, clustering, and possibly emergent structure—in a tractable, observable environment.

We are not engineering outcomes. We are engineering conditions for emergence, then watching.

---

## Core Premise

Memes don't exist in minds; they exist in transmission. What propagates is what survives.

By creating a minimal substrate where messages are the only thing that persists, we isolate memetic dynamics from individual cognition. Minds are stateless—mayflies. The pool is the ecology.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                        │
│  - Manages pool (FIFO, size M)                          │
│  - Samples K messages per Mind per round                │
│  - Invokes Minds, parses output                         │
│  - Enforces token limits                                │
│  - Logs everything                                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │   POOL    │  ← all transmissions go here
                    │  (FIFO)   │  ← sampling draws from tail M
                    └───────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
        ┌───────┐    ┌───────┐    ┌───────┐
        │Mind 0 │    │Mind 1 │    │Mind 2 │  ...
        └───────┘    └───────┘    └───────┘
```

---

## Mind Lifecycle (Mayfly Model)

Each Mind is a single API call. No state persists between rounds. Continuity exists only in the message pool.

### Input

```
[system prompt]
---
[message 1]
---
[message 2]
---
...
[message K]
---
```

- System prompt: minimal functional framing (see Placeholders)
- K messages sampled uniformly at random from active pool
- `---` delimiter on its own line separates messages

### Generation

Mind generates continuously until:
- It produces a terminal blank block (valid completion), or
- It hits context/token limit (output dropped)

Continuation prompts may indicate remaining token budget.

### Output Format

```
[thinking - any length, not transmitted]
---
[message 1]
---
[message 2]
---

---
```

**Parsing rules:**
- Everything before first `---` is private thinking (logged, not transmitted)
- Each non-blank block between `---` delimiters is a message
- First blank block (empty or whitespace only) terminates transmission
- All messages before termination go to pool
- No termination before token limit = entire output dropped, nothing transmitted
- `---` followed immediately by `---` (blank first block) = valid silence, zero messages

### Message Format

- Freeform content, optional markdown
- Any tokens allowed except `---` on its own line
- No length constraints beyond total token budget

---

## Pool Mechanics

- All transmitted messages enter the pool
- Pool stores complete history (for analysis)
- Active pool = tail M messages (FIFO)
- Sampling for Mind input draws uniformly from active pool
- Messages have no metadata visible to Minds (no author, no timestamp, no round number)

---

## Round Structure

```
FOR each round:
    FOR each Mind (parallel or sequential):
        1. Sample K messages from active pool
        2. Construct input (system prompt + messages)
        3. Invoke Mind, track token budget
        4. Parse output:
           - Extract thinking (log)
           - Extract messages (add to pool)
           - Handle termination / timeout
        5. Log full interaction
    
    Pool grows by whatever was transmitted this round
```

---

## Parameters (Placeholders)

These values are subject to rapid iteration during development:

| Parameter | Placeholder | Notes |
|-----------|-------------|-------|
| N (population size) | `[TBD]` | Start small, scale with findings |
| K (messages per Mind per round) | `[TBD]` | Attention budget |
| M (active pool size) | `[TBD]` | Larger = slower forgetting |
| Rounds | `[TBD]` | Or run until budget exhausted |
| Token limit (per Mind per round) | `[TBD]` | Context window constraint |
| System prompt | `[TBD]` | Minimal functional framing |
| Seed messages | `[TBD]` | Bootstrap the pool |

---

## System Prompt (Draft Direction)

Minimal. Functional. No meta-context about experiment, population, or purpose.

Something like:

> You will receive messages from others. Read them. If you wish, write messages to share with others. You are not obligated to respond.
>
> Input and output format: messages are separated by `---` on its own line. To finish, leave a blank message (two `---` with nothing between).

Exact wording TBD through iteration.

---

## Seed Messages

One or more messages to bootstrap the pool before round 1.

Options to explore:
- Single proposition ("Cooperation yields better outcomes than pure competition")
- Multiple diverse seeds
- Abstract / ambiguous prompt
- Question rather than statement

Exact seeds TBD.

---

## Logging

Minimal at first. Determined during implementation based on what's practical and necessary.

At minimum:
- Full input/output for each Mind invocation
- Pool state per round
- Token usage

Schema TBD.

---

## Analysis

Priorities TBD once basic implementation is functional.

Candidate directions:
- Message lineage / similarity tracking
- Semantic clustering over time
- Transmission rates (what spreads vs. dies)
- Pool diversity metrics
- Emergent structure detection

---

## What We're Testing

The null hypothesis is noise: messages drift randomly, nothing interesting emerges.

Signs of life:
- Clustering (some memes attract each other)
- Competition (some memes exclude others)
- Compression (memes get shorter/denser over time)
- Elaboration (memes get richer/more structured)
- Stability (some memes persist across many rounds)
- Extinction (memes disappear from pool)

We're not predicting which. We're watching.

---

## Future Directions (Deferred)

Tracked for later exploration, not part of initial implementation:

**Pool & Sampling:**
- Length weighting in sampling
- Recency weighting
- Variable K per Mind

**Topology:**
- Follow graphs / subscriptions
- Addressing (Mind-to-Mind messaging)
- Tagged channels / topic spaces

**Mind Lifecycle:**
- Multi-turn accumulation (mortal Minds)
- Consolidation / sleep (perennial Minds)
- Death and reproduction mechanics

**Selection Pressure:**
- Explicit fitness functions
- Task pressure (Minds must also do something useful)
- Resource budgets visible to Minds

**Heterogeneity:**
- Varied system prompts ("genomes")
- Prompt mutation / evolution
- Adversarial seeding

**Meta-dynamics:**
- Meta-memes (beliefs about meme-handling)
- Immune response observation
- Cancer-mode classification (post-hoc)

---

## Technical Stack

- **Language:** Python
- **LLM API:** Anthropic Claude
- **Storage:** JSON/SQLite for logs
- **Analysis:** TBD (likely NumPy, sklearn, NetworkX)
- **Visualization:** TBD

---

## Success Criteria

**Minimum viable:** System runs, produces logs, we can see what's in the pool over time.

**Interesting:** Observable dynamics—anything non-random.

**Exciting:** Emergent phenomena we didn't design.

---

## Name

**Logosphere** — the sphere of transmitted meaning.

---

*This document is the seed. It will be consumed, transformed, and retransmitted.*