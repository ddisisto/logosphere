# RFC: Recall Sampling

## Status
Draft — concept exploration

## Problem

Dialogue-driven thought generation exerts strong pressure on pool composition. User messages and the drafting process produce bursts of semantically related thoughts that can dominate the active pool, displacing prior clusters.

Under current FIFO rotation with uniform random sampling:
- Active clusters self-reinforce through repeated sampling
- Dormant clusters receive no exposure once members rotate out
- Historical depth is inaccessible — consolidated knowledge can go fully extinct
- No mechanism exists for spontaneous recall of relevant-but-dormant material

The system has working memory but no long-term memory retrieval.

## Observation

Clusters accumulate historical weight (total members across all time) that diverges from active presence (members within FIFO window). A cluster with historical weight 50 but active presence 2 represents consolidated knowledge at risk of extinction.

When a primary sample surfaces from such a cluster, it's a retrieval cue — potentially the last opportunity to access that lineage before complete displacement.

## Proposal

Introduce **recall sampling**: secondary sampling triggered by primary samples, pulling additional thoughts from the same cluster's history beyond the FIFO horizon.

### Mechanism

1. Primary sampling proceeds as current implementation (uniform random from active pool)
2. For each primary sample, evaluate cluster state:
   - Active presence (size within FIFO)
   - Age/position (proximity to displacement boundary)
   - Historical depth available
3. Based on these signals, optionally pull additional samples from cluster history
4. Interleave recall samples with their triggering primary sample
5. Recall samples displace other primary samples to maintain context budget

### Triggering Heuristics (to be refined)

Candidates for recall activation:
- **Rarity**: low active presence relative to historical weight
- **Position**: primary sample near FIFO boundary (final chances)
- **Depth available**: cluster has substantial history beyond active pool

The rarer and more endangered the cluster, the more context it earns when surfaced.

### Context Presentation

Recall samples appear interleaved with their cluster, flagged but using consistent metadata:

```yaml
thinking_pool:
  - |  # age: 47, cluster: {id: 3, size: 2}
    primary sample from sparse cluster
  - |  # age: 312, cluster: {id: 3, size: 2}, recall
    historical member, pulled by association
  - |  # age: 287, cluster: {id: 3, size: 2}, recall
    another historical member
  - |  # age: 8, cluster: {id: 11, size: 14}
    primary sample from active cluster (no recall needed)
```

Key observations the mind can make:
- Cluster size (2) accounts for active members only
- More samples than size indicates recall from history
- Age gap between primary and recall shows temporal depth
- Active clusters get breadth; sparse clusters get depth

### Asymmetry

Active clusters receive many independent samples across iterations (breadth). Dormant clusters receive few samples, but each comes with historical context (depth).

This is intentional:
- Active clusters are self-maintaining; they don't need historical reinforcement
- Dormant clusters earn resurrection through chance primary sampling
- The near-displacement position becomes high-leverage: last breath, loudest voice

## Motivation: Memory Analogy

The mechanism parallels human memory retrieval:

| Human Memory | Logosphere |
|--------------|------------|
| Decay / interference | FIFO rotation |
| Consolidation | Clustering |
| Task focus crowding working memory | Dialogue pressure |
| Retrieval cue | Primary sample from sparse cluster |
| Associative activation | Recall sampling |
| Memory trace strength | Historical weight |

Frequently rehearsed knowledge is recalled flatly, automatically. Half-forgotten material, accidentally triggered, comes with a flood of associated context. The surprise of recall carries more than the routine of familiarity.

## Properties

**Associative, not protective.** Recall is triggered by what surfaces naturally, not by scanning for endangered clusters. Protection may emerge where useful, but isn't explicitly engineered.

**Budget-neutral.** Recall samples displace primary samples. Depth costs breadth. The total context allocation remains fixed.

**Resurrection pathway.** A fully dormant cluster (zero active presence) can still be reached if a new thought clusters near its centroid. Centroid matching during incremental clustering can awaken dormant clusters, making their history available for recall.

**Observable to the mind.** The metadata reveals the mechanism. The mind can reason about what it's seeing: "this cluster is fading, I'm being shown its history, this may be final exposure."

## Open Questions

- Exact triggering thresholds (rarity ratio, position sensitivity)
- Recall budget per cluster (how many historical samples to pull)
- Selection within history (nearest to centroid? most recent? random?)
- Interaction with existing display limits (chars, count)

## Future Extensions

Explicit protective sampling (surfacing endangered clusters independent of primary sampling) could use similar infrastructure but different triggering logic. Deferred pending observation of whether associative recall provides sufficient protection organically.

---

*This RFC captures concept and motivation. Implementation details to follow if path is pursued.*
