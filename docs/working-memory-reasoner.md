# Working Memory Pool: Project Plan

*Reasoning as memetic ecology*

---

## Core Philosophy

**No explicit protocols.** The pool state *is* the output.

- No `[ANSWER]` tags - memes don't need to declare themselves answers
- No hardcoded marker weights - if `[IMPORTANT]` helps, it wins by replication
- Termination is dynamics-based, not protocol-based
- We measure and steer the pool, not the answers

The reasoning isn't in the generation - it's in the pool. Thoughts compete for attention. Persistence requires transmission. What survives is what fit the selection pressure.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    REASONING LOOP                        │
│  while not terminated:                                   │
│    1. Sample K thoughts from pool                        │
│    2. Model thinks, transmits (no protocol)              │
│    3. Embed, add to pool                                 │
│    4. Measure: diversity, clusters, stability            │
│    5. Check termination (dynamics only)                  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │   POOL    │  ← working memory (VectorDB)
                    │  (FIFO)   │  ← tail M active
                    └───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   POOL STATE = OUTPUT │
              │   Sample from dominant │
              │   cluster or return    │
              │   trajectory           │
              └───────────────────────┘
```

---

## Termination Conditions (Dynamics Only)

No explicit signals. Terminate based on pool state:

| Condition | Detection | Output |
|-----------|-----------|--------|
| **Converged** | Single cluster > 50% pool, coherence > 0.75 | Sample from dominant cluster |
| **Multi-stable** | Multiple distinct clusters, stable for N iterations | Return cluster representatives |
| **Chaotic/Timeout** | Max iterations, no stability | Return trajectory for analysis |
| **Early monoculture** | Pool homogeneous within few iterations | Could have one-shot; note this |

---

## Metrics (Per Iteration)

Track these to understand and steer dynamics:

```python
metrics = {
    'diversity': mean_pairwise_distance(active_pool),  # Higher = more diverse
    'num_clusters': len(attractor_state['clusters']),
    'dominant_cluster_share': largest_cluster_size / pool_size,
    'coherence': dominant_cluster['coherence'],  # Intra-cluster similarity
    'stability': clusters_unchanged_for_n_iterations,
    'drift': distance_from_seed_centroid,  # How far from starting point
}
```

---

## System Prompt (Minimal)

No protocol instructions. Just framing:

```
You receive thoughts from a shared pool.

Read them. Think privately. Transmit thoughts worth keeping.

Transmitted thoughts persist and compete for attention.
Thoughts not transmitted are forgotten.

Format: Thoughts separated by --- on its own line.
```

That's it. No "when you have an answer", no "tag important things". Let structure emerge.

---

## Interventions (Steering Dynamics)

Interventions operate on the *pool*, not on *protocols*:

| Intervention | What It Does |
|--------------|--------------|
| **Mask attractors** | Exclude messages from known-bad clusters when sampling |
| **Inject diversity** | Add seeds from outside current basin |
| **Annealing** | Vary K (sample size) over iterations - high early, low late |
| **Counter-context** | Detect drift toward attractor, inject opposing seeds |
| **Temperature** | Vary randomness in sampling - explore vs exploit |

These steer dynamics without telling the model what to do.

---

## Implementation Status

### Done
- [x] VectorDB (replaces Pool)
- [x] EmbeddingClient
- [x] AttractorDetector (HDBSCAN clustering)
- [x] Basic reasoner loop
- [x] CLI entry point

### Now: Philosophy Pivot
- [ ] Remove `[ANSWER]` detection from reasoner
- [ ] Dynamics-only termination (convergence, stability, timeout)
- [ ] Add per-iteration metrics logging
- [ ] Update system prompt (remove protocol)
- [ ] Test: run to observe dynamics, not to get answers

### Later
- [ ] Intervention implementations (masking, injection, annealing)
- [ ] Compare: one-shot vs N-iteration pool
- [ ] Find problems where pool dynamics matter
- [ ] Small model experiments (cheap iteration)

---

## Key Design Decisions

### Pool State = Output

The "answer" is not extracted via protocol. It emerges:
- Converged pool → sample dominant cluster
- Multi-stable → return the clusters themselves (interesting!)
- Chaotic → the trajectory is the data

### No Marker Vocabulary

Don't design `[IMPORTANT]`, `[DISCARD]`, etc. If markers help, they'll emerge and win by replication. If they don't emerge, they weren't needed.

Observe what emerges. Analyze post-hoc. Don't prescribe.

### Dynamics Over Accuracy

Primary goal: **measure and steer pool dynamics**

Secondary: answer quality (just one signal among many)

This means:
- We might make answers *worse* with iteration - that's data
- Early convergence is boring, not success
- Chaotic trajectories are interesting, not failures

---

## Open Questions

1. **Does iteration help or hurt?** Compare one-shot vs pool-after-N on same problems

2. **What emerges?** Run many iterations, analyze pool content for patterns

3. **Can we steer?** Test interventions - do they change dynamics measurably?

4. **Small vs large models** - Do small models show more interesting dynamics (more variance, more iterations needed)?

5. **Problem dependence** - Which problems benefit from pool? Which don't?

---

## Connection to Logosphere

This is Logosphere turned inward:

> Memes don't exist in minds; they exist in transmission.

Becomes:

> Thoughts don't exist in reasoning steps; they exist in pool transmission.

The pool is the selection pressure. Attention scarcity is the constraint. What cognitive structure arises when memes compete for working memory?

---

*The reasoning is in the transmission. The answer is in the pool state.*
