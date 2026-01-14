# Incremental Clustering Design

## Problem

The current `logos analyze` implementation re-clusters a sliding window each iteration, which:
- Assigns the same message to different clusters as the window moves
- Reports cumulative "message-iterations" not unique message counts
- Doesn't provide stable cluster membership for tracking

## Goals

1. **Stable assignment**: Each message assigned to exactly one cluster (or noise)
2. **Identity continuity**: Cluster identities persist across iterations
3. **Incremental processing**: O(new messages + recent noise) per iteration, not O(all messages)
4. **Local continuity**: Adjacent iterations have smooth cluster evolution (drift is fine)

## Design

### Data Model

```
ClusterRegistry:
  clusters: dict[cluster_id, ClusterState]

ClusterState:
  id: str                    # e.g., "cluster_0"
  centroid: np.ndarray       # Current centroid (evolves with new members)
  member_count: int          # Total assigned messages
  created_at: int            # Iteration when spawned
  last_active: int           # Last iteration with new assignment
  coherence: float           # Intra-cluster similarity (optional tracking)

AssignmentTable:
  assignments: dict[message_id, cluster_id | "noise" | "fossil"]
  noise_iteration: dict[message_id, int]  # When message was marked noise

SnapshotHistory (optional, for trajectory analysis):
  snapshots: list[ClusterSnapshot]  # Per-iteration state for visualization
```

### Core Algorithm

Each iteration:

```
def process_iteration(new_messages, cluster_registry, assignment_table, config):
    N = config.noise_reconsider_iterations  # e.g., 20
    threshold = config.centroid_match_threshold  # e.g., 0.3 cosine distance
    min_cluster_size = config.min_cluster_size  # e.g., 3

    # 1. Gather candidates
    candidates = new_messages + get_recent_noise(assignment_table, current_iteration - N)

    # 2. Phase 1: Match against existing clusters (including dormant)
    unmatched = []
    for msg in candidates:
        best_cluster, distance = find_nearest_cluster(msg.embedding, cluster_registry)

        if best_cluster and distance < threshold:
            # Assign to existing cluster
            assign(msg, best_cluster, assignment_table)
            update_centroid(best_cluster, msg.embedding)
        else:
            unmatched.append(msg)

    # 3. Phase 2: Discover new clusters from unmatched
    if len(unmatched) >= min_cluster_size:
        labels = hdbscan.fit_predict([m.embedding for m in unmatched])

        for label in unique_labels(labels):
            if label == -1:
                continue  # noise handled below

            cluster_members = [m for m, l in zip(unmatched, labels) if l == label]
            new_cluster = spawn_cluster(cluster_members, cluster_registry)

            for msg in cluster_members:
                assign(msg, new_cluster, assignment_table)
                unmatched.remove(msg)

    # 4. Mark remaining as noise
    for msg in unmatched:
        mark_noise(msg, current_iteration, assignment_table)

    # 5. Fossilize old noise
    fossilize_old_noise(assignment_table, current_iteration - N)
```

### Centroid Update

When a message is assigned to a cluster:

```
def update_centroid(cluster, new_embedding):
    n = cluster.member_count
    cluster.centroid = (cluster.centroid * n + new_embedding) / (n + 1)
    cluster.member_count += 1
    cluster.last_active = current_iteration
```

This allows clusters to drift over time as new content shapes them.

### Finding Nearest Cluster

```
def find_nearest_cluster(embedding, cluster_registry):
    best_cluster = None
    best_distance = float('inf')

    for cluster in cluster_registry.clusters.values():
        distance = cosine_distance(embedding, cluster.centroid)
        if distance < best_distance:
            best_distance = distance
            best_cluster = cluster

    return best_cluster, best_distance
```

All clusters are considered, including dormant ones. A lone message can join a dormant cluster if within threshold (no min_cluster_size requirement for existing clusters).

### Noise Lifecycle

```
New message arrives
    ↓
Check against existing centroids
    ↓
No match within threshold
    ↓
Goes to HDBSCAN pool
    ↓
HDBSCAN marks as noise (no cluster formed)
    ↓
Marked as noise with current iteration timestamp
    ↓
Reconsidered for N iterations (re-enters candidate pool each iteration)
    ↓
After N iterations: fossilized (permanent noise, stops being reconsidered)
```

## Key Properties

### Why Two Phases?

**Phase 1 (centroid matching)** handles:
- Assignment to existing clusters (including dormant)
- No minimum size requirement - single messages can join existing clusters
- Drift absorption - old noise matches clusters that moved into range

**Phase 2 (HDBSCAN)** handles:
- Discovery of genuinely new clusters
- Requires min_cluster_size to avoid spurious clusters
- Only runs on messages that don't match existing clusters

### Drift and Continuity

- Centroids update incrementally, so adjacent iterations have similar positions
- Drift velocity trackable: `||centroid_t - centroid_{t-1}||`
- Cluster identity is continuous even as semantic position evolves
- "Local continuity" preserved; global position may drift significantly over time

### Dormant Clusters

- Clusters with no recent assignments remain in registry with frozen centroid
- Can be reactivated if new content matches (phase 1 checks all clusters)
- "Fossil mode": dormant clusters persist indefinitely
- Future option: archive after M iterations dormant

## Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `centroid_match_threshold` | 0.3 | Max cosine distance for phase 1 matching |
| `min_cluster_size` | 3 | HDBSCAN threshold for new cluster formation |
| `noise_reconsider_iterations` | 20 | How long noise stays in candidate pool |

## Migration

For existing sessions:
1. Run initial full HDBSCAN on all messages to bootstrap cluster registry
2. Assign all messages based on initial clustering
3. Future iterations use incremental algorithm

Or start fresh:
1. Empty cluster registry
2. First iteration: all messages go to HDBSCAN, clusters emerge naturally
3. Subsequent iterations: incremental

## Analysis Output

With stable assignments, `logos analyze` can report:
- Unique message counts per cluster (not cumulative iterations)
- Cluster trajectories (centroid position over time)
- Drift velocity, convergence/divergence between clusters
- Dormancy periods, reactivation events

## Future Extensions

- **Cluster splitting**: If coherence drops below threshold, consider splitting
- **Dampened updates**: Exponential moving average instead of true mean
- **Soft assignment**: Track membership probability, allow reassignment if confidence drops
- **Cross-branch analysis**: Compare cluster evolution across branches
- **Cluster shape metrics**: Track per-cluster geometry via covariance eigenvalues - aspect ratio (elongation), effective dimensionality. Could surface tight vs diffuse clusters, shape evolution over time, candidates for splitting
