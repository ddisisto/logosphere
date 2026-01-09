# Embedding Observability: Design Sketch

*What's interesting to look at in the embedding space?*

---

## Single Branch Analysis

### 1. Cluster Dynamics

**What to track:**
- Cluster count per iteration
- Cluster birth/death/merge events
- Dominant cluster trajectory (which one "wins"?)
- Per-cluster coherence over time (getting tighter or looser?)

**Questions:**
- When does convergence start? (phase transition detection)
- Are there stable multi-attractor states or does one always dominate?
- What predicts which cluster will win?

### 2. Semantic Drift

**Metrics:**
- Distance from seed centroid over time
- Pool centroid velocity (how fast is meaning shifting?)
- Drift direction (toward what?)

**Questions:**
- Does drift stabilize or continue indefinitely?
- Is drift direction predictable from seed content?

### 3. Diversity

**Metrics:**
- Mean pairwise cosine distance in active pool
- Embedding space coverage (convex hull volume? spread?)
- Entropy of cluster assignments

**Questions:**
- Does diversity monotonically decrease or oscillate?
- What's the "floor" - minimum sustainable diversity?

### 4. Attractor Detection

**Phenomena:**
- Fixed points: messages that keep getting regenerated verbatim
- Near-attractors: semantic region that captures and holds messages
- Basin depth: how many iterations to escape if perturbed?

**Questions:**
- Can we predict attractor formation early?
- What makes some attractors "stickier" than others?

### 5. Message Lineage

**Using sampled_ids field:**
- Build influence graph: message A sampled → message B generated
- Identify "super-spreaders" (messages sampled disproportionately)
- Track mutation chains (how ideas evolve through generations)

**Questions:**
- Do influential messages have identifiable properties?
- How many "generations" until a seed idea is unrecognizable?

### 6. Injection Impact Analysis

**For each injection:**
- Before/after cluster structure comparison
- New cluster formation vs. absorption into existing
- Propagation radius (how many generations show influence?)
- Decay curve (how quickly does injection signal fade?)

---

## Cross-Branch Analysis

### 1. Divergence Metrics

**Track over time:**
- Centroid distance between branches
- Cluster overlap (Jaccard of cluster assignments?)
- Vocabulary/topic divergence

**Questions:**
- When does meaningful divergence begin?
- Is divergence monotonic or can branches re-converge?

### 2. Counterfactual Analysis

**Compare treatment vs. control:**
- "What changed because of intervention X?"
- Isolate causal impact from natural drift
- Statistical significance of differences

### 3. Attractor Comparison

**Questions:**
- Do branches converge to same attractors via different paths?
- Are attractors properties of the seed, the model, or emergent?
- Can we identify "universal" vs "contingent" attractors?

---

## Potential Visualizations

### 2D Projection (UMAP/t-SNE)
- Animate over iterations
- Color by: round, cluster, branch, injection-influence
- Show text on hover
- Trace individual message trajectories

### Cluster Timeline (Swimlane)
```
Cluster A: ████████████████░░░░░░░░░░  (dominant early)
Cluster B: ░░░░████████████████████████  (takes over)
Cluster C: ░░░░░░░░████░░░░░░░░░░░░░░░░  (transient)
           |----|----|----|----|----|----|
           0   50  100  150  200  250  300  iteration
```

### Influence Graph
- Nodes = messages
- Edges = sampled_ids links
- Layout by embedding similarity
- Size by "influence" (times sampled, offspring count)

### Drift Plot
- X: iteration
- Y: distance from seed centroid
- Overlay: diversity, cluster count

### Branch Comparison
- Side-by-side embedding projections
- Difference heatmap
- Divergence curve over shared history

---

## Data Requirements

Already have:
- `embeddings.npy` - full embedding matrix
- `metadata.jsonl` - round, branch, sampled_ids per message

May need:
- Cluster assignments per iteration (compute on demand or cache?)
- Pre-computed metrics time series (for fast dashboard loading)

---

## Implementation Phases

**Phase 1: CLI metrics**
- `logos analyze` command
- Compute and print: diversity, cluster count, drift from seed
- Per-iteration CSV export for external plotting

**Phase 2: Static visualizations**
- Generate PNG/SVG plots
- UMAP projection with cluster coloring
- Drift/diversity over time charts

**Phase 3: Interactive dashboard**
- Web UI for exploration
- Animated timeline
- Click-to-inspect messages
- Branch comparison view

---

## Open Questions

1. **Computational cost**: HDBSCAN on every iteration? Or sample checkpoints?

2. **Baseline establishment**: What's "normal" diversity/drift for a healthy pool?

3. **Intervention detection**: Can we automatically detect when something interesting happened?

4. **Semantic grounding**: How to relate embedding clusters to human-readable topics?

5. **Real-time vs post-hoc**: Build for live monitoring or batch analysis?
