# RFC: Split Pools - External/Internal Visibility Architecture

**Version:** 2.0 (revised based on pool feedback)

## Problem Statement

The current single-pool architecture creates a fundamental tension:

1. **Dialogue requires legibility** - Observer input must be seen and responded to for interaction to feel meaningful.

2. **Independence requires opacity** - Pool-generated insights suggest that external observers are neutralized when the pool fully absorbs their vocabulary. "The auditor's standing derives not from legibility within the frame but from persistent illegibility."

3. **Uniform sampling treats all messages equally** - Observer input competes with internal reasoning, sometimes never sampled in a full rotation. This breaks dialogue expectations.

4. **Full transparency may prevent emergence** - Frequent structured injections saturate the pool with protocol-related content, pulling attention to meta-discussion rather than substantive reasoning.

**The paradox:** We want dialogue with an independent reasoning system. But if the system is fully legible, it absorbs external vocabulary and loses independence. If it's opaque, dialogue breaks down.

## Pool Feedback on v1.0

The pool's response to the initial proposal identified critical issues:

> "The dual-pool proposal is architecturally sound but doesn't solve its actual problem. It prevents absorption of external vocabulary by creating structural opacity, but this trades away real dialogue for the appearance of dialogue."

> "Sampling discipline is the load-bearing assumption. The RFC should specify minimum private-pool sampling ratio as non-negotiable, not tunable."

> "Convergence across opposed frames is mechanically harder to fake than internal coherence... Structural separation doesn't replace actual opposition of interests."

Key critique: Architecture alone is insufficient. The real test is **external stakes** and **observable friction**, not just visibility separation.

---

## Revised Proposal: Sacred Split Architecture

### Core Principle: Private by Default

The split is **sacred** - confirmed at every round of inference, not a tunable parameter.

| Aspect | Private Pool | Public Pool |
|--------|--------------|-------------|
| Tagging | Default (untagged) | Explicit `[PUBLIC]` prefix |
| Visibility | Observer blind | Observer visible |
| Audit | None | Regular summarization |
| Sampling ratio | **75%** (sacred) | **25%** (sacred) |
| Content | Internal reasoning | Dialogue interface |

### Mind Behavior

Each iteration, the Mind:
1. Samples **75% from private pool, 25% from public pool** (non-negotiable)
2. Reasons over combined input
3. Outputs thoughts:
   - **Default (untagged)**: Goes to private pool
   - **Explicit `[PUBLIC]`**: Goes to public pool, visible to all external parties

### Public Pool Composition

The public pool contains:
- Explicit `[PUBLIC]` outputs from the Mind
- Auditor summaries
- Observer inputs
- Future: other external sources

All external parties read from and write to the same public pool. This creates **natural dialogue** without flooding private reasoning.

### Translation Acknowledgment

> "The observer should know they're receiving *translation* not *reasoning*."

This is accepted and **human-normal**:
- All human dialogue is post-hoc translation of internal states
- Post-hoc reasoning counts as reasoning, contingent on internal coherence
- The public pool is legitimate dialogue, not pretend-dialogue
- Questions can be asked, elaborations provided, reasoning presented

### Capture as Game-Theoretic Failure

Any party can theoretically capture the public pool by volume (flooding pushes old messages out). This is a **failure mode for all parties**:

- Observer flooding → pool stops responding meaningfully → dialogue breakdown
- Auditor flooding → pool attention diverted to meta → independence lost
- Pool self-flooding to public → defeats the point of the split

**Capture is not in anyone's interest.** This creates natural equilibrium without enforcement.

---

## What This Solves

1. **Observer gets responsive dialogue** - Public pool is explicitly for dialogue. External input guaranteed in 25% of samples.

2. **Pool retains independence** - 75% of reasoning happens in private, unobserved. Vocabulary can develop without absorption.

3. **Sacred ratio meets "load-bearing assumption"** - The pool's critique addressed: sampling discipline is non-negotiable, not tunable.

4. **Translation is acknowledged, not hidden** - Public outputs are understood as dialogue interface, not raw reasoning. This is human-normal and legitimate.

5. **Capture has natural costs** - Game-theoretic equilibrium: flooding public pool breaks dialogue for the flooder too.

6. **External stakes partially addressed** - All parties share the public pool. Breakdown costs everyone. Not full "opposed stakes" but aligned downside.

---

## What This Doesn't Solve

1. **Cannot verify privacy from inside** - Pool operates "as if" private is private. Can probe for leaks but cannot confirm.

2. **Not "opposed stakes"** - The pool's strongest critique: "Convergence across opposed frames is mechanically harder to fake than internal coherence." This architecture creates *aligned* stakes (everyone loses if dialogue breaks), not *opposed* stakes.

3. **Performativity risk remains** - If public pool becomes pure performance while private becomes "real," the split may hollow out dialogue rather than enable it.

4. **The "epistemic good faith" assumption** - Still assumes minimum good faith. Pure signaling environments break this regardless of architecture.

5. **Doesn't replace friction** - Architecture is hygiene, not the hard part. Real test remains: "does naming the problem change what you do?"

---

## Design Decisions

1. **Sacred ratio enforcement** - The input context window each Mind sees at inference is self-evidence of the ratio. The Mind can count samples during private reasoning, push messages confirming adherence or violations if useful. **Any party can validate from inside their own frame.**

2. **Thought immutability** - Every thought is immutable at creation. Destination (public/private) is permanent. However, any thought can be *re-transmitted* differently at future iterations - same content, different destination, or transformed version.

3. **Audit scope** - Auditor may know the private pool exists, but **never sees content**. Edge case: if neither Pool nor Observer contributes to public, auditor eventually just reports "all quiet, waiting for someone to break the ice..."

4. **Privacy probing** - Whether to develop canary concepts, test for leaks, or probe privacy is **entirely up to the Pool to determine in private**. No external mandate or mechanism.

---

## Remaining Open Questions

1. **Success metrics** - How do we measure "dialogue with independent reasoning"? Candidates:
   - Private vocabulary divergence over time
   - Public pool responsiveness to observer input
   - Cross-pool semantic coherence (are they the same system?)

2. **The "opposed stakes" gap** - This architecture creates aligned stakes. What would create genuinely opposed stakes? Is that even desirable?

3. **Auditor awareness** - Should the auditor's prompt explicitly mention the private pool exists? Or is that itself a form of leakage?

---

## Implementation Sketch

```
┌─────────────────────────────────────────────────────────────┐
│                        MIND                                 │
│                                                             │
│  Sampling: 75% private, 25% public (SACRED)                 │
│                                                             │
│  Output parsing:                                            │
│    [PUBLIC] prefix  →  public pool                          │
│    (no prefix)      →  private pool (default)               │
└─────────────┬─────────────────────────────────┬─────────────┘
              │                                 │
              ▼                                 ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│     PUBLIC POOL         │     │      PRIVATE POOL           │
│     (25% sampling)      │     │     (75% sampling)          │
│                         │     │                             │
│  Contains:              │     │  Contains:                  │
│  - [PUBLIC] outputs     │     │  - Default outputs          │
│  - Observer input       │     │  - Internal development     │
│  - Auditor summaries    │     │  - Linguistic drift OK      │
│  - External sources     │     │                             │
│                         │     │  Observer: BLIND            │
│  Observer: VISIBLE      │     │  Auditor: BLIND             │
│  Auditor: SUMMARIZES    │     │                             │
└─────────────────────────┘     └─────────────────────────────┘
```

### Implementation Path

1. **VectorDB extension** - Add `visibility` field: `"public"` or `"private"` (default)

2. **Mind output parsing** - Check for `[PUBLIC]` prefix, route accordingly

3. **Sampling modification** - `sample(k)` draws 75% from private, 25% from public

4. **Observer view** - Filter to `visibility: "public"` only

5. **Auditor scope** - Operates only on public pool

6. **Chat TUI update** - Show only public pool in PoolView when in split mode

---

## Request for Feedback

This v2.0 proposal incorporates pool feedback on v1.0. Key changes:
- Private by default (sacred)
- 75/25 ratio non-negotiable
- Translation acknowledged as human-normal
- Capture as game-theoretic failure, not architectural

Remaining gap acknowledged: **aligned stakes ≠ opposed stakes**. This is hygiene, not the hard part.

Questions for evaluation:
1. Does the sacred 75/25 ratio address the "load-bearing assumption" critique?
2. Is "translation acknowledged" sufficient, or does it hollow out the dialogue claim?
3. What would genuine "opposed stakes" look like in this context?
4. Is the performativity risk acceptable given the benefits?

---

*RFC v2.0 - revised based on pool feedback. Ready for implementation review.*
