# RFC: Narrowing Draft Function

## Problem

The draft buffer currently serves three distinct purposes:

1. **Response presentation** - user-facing drafts for acceptance
2. **Working space** - sequential accumulation during user absence
3. **Explicit signalling** - hard signals (skip), soft signals (+1)

This overloading creates confusion at inference time:
- What should a draft contain under what conditions?
- Observed behaviors: immediate complete draft, then mix of +1, similar rewrites, different angles, incremental edits (not self-contained), or nothing
- The mind must understand signalling mechanics to use them effectively
- Lack of clarity apparent in actual usage

## Goal

Narrow drafts to a single function: **user-facing response presentation**.

Move the other functions elsewhere:
- Working space → thinking pool (via presentation change)
- Explicit signalling → implicit signal extraction (system-derived)

## Proposed Changes

### 1. Split Thinking Pool Presentation

No new buffer. Same content, same mechanics. Different *view* presented:

```yaml
thinking_pool:
  recent:  # sequential, oldest-first, own char budget, no cluster metadata
    - |  # age: 4
      approaching cut-off - consolidate or re-append
    - |  # age: 3
      building toward something...
    - |  # age: 2
      refinement
    - |  # age: 1
      latest addition

  sampled:  # random from older pool, existing budget
    - |  # age: 28, cluster: {id: 3, size: 8}
      surfaced by chance
    - |  # age: 51, cluster: {~}
      older noise
```

**Recent section:**
- Own char budget (separate from sampled)
- Ordered oldest-first (top = approaching boundary, needs attention)
- No cluster metadata (too fresh)
- Provides visibility of "what you just thought"

**Effects:**
- Reduces blind repetition (see that point X was just raised)
- Enables sequential building during processing
- Encourages diversity (recent covers X, sampling might surface Y)
- Reinforcement still works via intentional re-emission

**Emergent behaviors to encourage:**
- Re-append important thoughts as they approach cut-off (maintains visibility + reinforces)
- Consolidate sequential chains into standalone summaries before they fragment

**Orphan reference concern:** Sequential thoughts may reference each other; when chain breaks, references orphan. Considered acceptable - the *process* (mode of thinking) is remembered even if specific context fades. Consolidation guidance mitigates.

### 2. Implicit Signal Extraction

Remove explicit signalling mechanics (skip, +1, consecutive_hard tracking).

Instead, derive stopping conditions from observable metrics:

| Metric | What it measures | Signal |
|--------|------------------|--------|
| Draft stability | Semantic similarity between consecutive drafts | High = converged |
| Thought production rate | New thoughts per iteration | Declining = exhausted |
| Cluster dispersion | How many clusters new thoughts land in | Low = narrowing focus |
| Draft-to-thought ratio | Drafts vs thinking-only iterations | High = ready to respond |
| Embedding trajectory | Direction of recent thoughts in space | Stabilizing = settling |

**Benefits:**
- Removes cognitive load at inference time
- More adaptable (tune metrics without changing protocol)
- Potentially more accurate (patterns reveal readiness)

**The mind still sees** relevant state (metrics or derivatives) - dynamics remain observable and reason-about-able. But explicit signal production not required.

**Stop conditions derived from:**
- "Ready" - converged, stable drafts, integrated response likely
- "Stuck" - cycling, no progress, needs input

Both trigger stops, but for different reasons. No "demand user attention" framing - just "nothing more to contribute right now."

### 3. Draft: New Framing and Function

*Outstanding - needs further discussion.*

With working space moved to thinking pool and signalling made implicit, drafts become purely:
- User-facing response candidates
- Self-contained, polished
- Produced when transitioning from processing to framing mode

Open questions:
- What triggers the transition to framing mode?
- Should draft be singular (current best) or accumulating?
- How does presence state interact with draft production?

## Risk: Problem Shifts to Thoughts

The multi-purpose confusion may just migrate from drafts to thoughts:
- Recent section = working space
- Sampled section = persistent memory
- Are these actually different enough?

Initial assessment: this is *less* problematic context, possibly even positive tension. The thinking pool is internal; confusion there is lower stakes than confusion in user-facing drafts. But complete separation could be explored if needed.

## Dialogue Round Lifecycle

```
USER INPUT
    ↓
[PROCESSING] ←── thinking accumulates, noise → clusters, draft shapes response
    ↓ start conditions met
[DRAFTING] ←── draft visible to user, refine as needed
    ↓ user accepts
USER INPUT (loop)
```

No IDLE state - acceptance immediately enables next user input.

### Start Conditions (PROCESSING → DRAFTING)

Transition when ready to present to user:

| Condition | Measurement | Rationale |
|-----------|-------------|-----------|
| Minimum iters | Counter since input | Cooling period, allow integration |
| Noise ratio | Unclustered / recent thoughts | Below threshold = input integrated |
| User presence | engaged/reviewing/absent | Modifies thresholds (engaged = faster) |

### Stop Conditions (halt iteration)

| Condition | Measurement | Rationale |
|-----------|-------------|-----------|
| Draft stable | No new draft for N iters | Converged |
| Thoughts exhausted | Production declining | Nothing more to add |
| Hard ceiling | Max iters per round | Safety valve |

### Metrics (Simplified)

| Metric | Measurement | Notes |
|--------|-------------|-------|
| Iters since input | Counter | Start threshold |
| Noise ratio | Unclustered / total recent | Integration progress |
| Draft changed | Boolean per iter | Stability signal |
| Thought production | Count per iter | Exhaustion signal |
| User presence | State enum | Threshold modifier |

No semantic analysis needed. Draft stability = "new draft emitted" vs "no new draft."

### Draft During Processing

Draft buffer available in PROCESSING phase, but:
- **Not user-visible** until DRAFTING phase
- **Purpose**: capture response shape, offload dialogue-specific content from thoughts
- **Character**: sketchy, incomplete ok - shaping, not presenting

This is intentional overloading: both uses are about the response, just at different readiness stages. Thoughts remain cross-context and persistent; draft (even during processing) is dialogue-specific and ephemeral.

**Open question**: Does mind need to know which phase?
- Option 1: Implicit - mind outputs `draft:`, system handles visibility
- Option 2: Explicit - mode indicator in input, mind adjusts expectations
- Leaning toward option 2 for clarity without adding output complexity

### User Presence Interaction

Presence modifies thresholds, doesn't hard-gate:

| Presence | Start threshold | Stop conditions |
|----------|-----------------|-----------------|
| Engaged | Lower (faster to draft) | Tighter (less iteration) |
| Reviewing | Balanced | Balanced |
| Absent | Higher (more processing) | Looser (more refinement ok) |

Presence can change mid-round; thresholds adjust dynamically. No hard resets.

## Open Questions

1. **Mode visibility**: Should mind see explicit phase indicator? (leaning yes)
2. **Threshold tuning**: What are sensible defaults for start/stop conditions?
3. **Noise ratio calculation**: Recent thoughts only, or include sampled?
4. **Draft shaping guidance**: How to frame "capture shape, not present" in protocol?

## Implementation Scope

Phase 1: Thinking pool split
- [ ] Split presentation (recent/sampled sections)
- [ ] Add char budget for recent section to config
- [ ] Update sampler: deterministic for recent, random for older
- [ ] Update input format with `recent:` / `sampled:` structure

Phase 2: Metrics and lifecycle
- [ ] Track noise ratio (unclustered / recent thoughts)
- [ ] Track iters since last user input
- [ ] Implement start condition evaluation
- [ ] Implement stop condition evaluation
- [ ] Add phase indicator to input (PROCESSING/DRAFTING)

Phase 3: Draft visibility
- [ ] Separate draft storage from draft presentation
- [ ] Drafts during PROCESSING stored but not shown
- [ ] Transition to DRAFTING makes drafts visible
- [ ] Mind sees phase in input, adjusts accordingly

Phase 4: Protocol update
- [ ] Reframe draft function in system prompt
- [ ] Remove explicit signal mechanics (+1, skip, consecutive_hard)
- [ ] Add guidance for draft-during-processing (shaping)
- [ ] Test and iterate with actual sessions
