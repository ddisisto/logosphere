# CLAUDE.md - Logosphere Architecture

## Experimental Boundary

**INSIDE THE EXPERIMENT:**
- Messages in the pool
- Mind inputs (sampled messages + system prompt)
- Mind outputs (parsed messages)

**OUTSIDE THE EXPERIMENT:**
- Runner (sampler, parser, embedder)
- API calls to LLM
- Analysis tools

**Critical:** Nothing inside the experiment can see round numbers, timestamps, authorship, or any metadata. Messages are anonymous and timeless from the perspective of Minds.

---

## Current Structure

```
logosphere/
├── src/
│   ├── core/
│   │   ├── vector_db.py       # Message storage with embeddings
│   │   ├── session.py         # Linear session management
│   │   ├── intervention_log.py # Append-only audit trail
│   │   ├── embedding_client.py # OpenRouter embedding API
│   │   └── mind.py            # LLM invocation and parsing
│   ├── logos/
│   │   ├── config.py          # Configuration defaults
│   │   ├── runner.py          # Core loop: sample → mind → embed → add
│   │   ├── analyze.py         # Sliding window cluster analysis (legacy)
│   │   └── clustering/        # Incremental clustering package
│   │       ├── models.py      # ClusterState, AssignmentTable
│   │       ├── algorithm.py   # Process iteration, bootstrap
│   │       └── manager.py     # ClusterManager persistence
│   ├── analysis/              # Standalone analysis tools
│   ├── exchange/              # Auditor hooks (experimental)
│   └── tui/                   # Chat TUI interface
├── scripts/
│   ├── logos.py               # Main CLI
│   └── extract_session.py     # Session extraction/forking utility
└── docs/                      # Design docs and sketches
```

---

## Development Setup

```bash
uv sync                      # Install core dependencies
uv sync --extra analysis     # Add analysis tools (hdbscan, numpy)
```

Run logos: `python scripts/logos.py --help`
Run tests: `pytest`

---

## Logos CLI

The main interface for running pool-based reasoning sessions.

### Session Management

```bash
logos init ./session "initial prompt"   # Create session
logos open ./session                    # Open existing
logos status                            # Current state
```

### Running Iterations

```bash
logos run 10                            # Batch iterations
logos step                              # Single iteration
logos inject "thought text"             # Add external message
```

### Configuration

```bash
logos config                            # Show current config
logos config --set model=anthropic/claude-haiku-4.5
logos config --set k_samples=10
```

### Analysis (Legacy)

```bash
logos analyze                           # Sliding window swimlane (legacy)
logos analyze --json                    # JSON output
logos analyze --from-iteration 10       # Analyze specific range
```

### Incremental Clustering

```bash
logos cluster status                    # Show cluster registry state
logos cluster bootstrap                 # Bootstrap from existing messages
logos cluster analyze                   # Analyze stable assignments
logos cluster show cluster_0            # Show members of a cluster
```

Clustering is integrated into `logos run` - new messages are assigned to clusters incrementally.

### Intervention Log

```bash
logos log                               # Show intervention history
```

---

## Core Concepts

### Session

A session is a directory containing:
- `session.json` - Iteration counter and config
- `vector_db/` - Embeddings and message metadata
- `clustering/` - Cluster registry and assignments (optional)
- `interventions.jsonl` - Audit log of all actions

Sessions are linear (no branching). Fork sessions by copying with `extract_session.py`.

### VectorDB

Append-only message storage:
- Each message has: text, embedding, round, mind_id
- Active pool = tail N messages
- Sampling is uniform random from active pool

### Mind

Stateless LLM invocation:
- Input: system prompt + K sampled messages
- Output: thinking (private) + transmitted messages (public)
- Parsing: `---` separates thoughts; first block is private

### Analysis (Legacy)

Sliding window cluster timeline:
- HDBSCAN clustering per iteration window
- ASCII swimlane visualization or JSON export
- Note: Same message may appear in different clusters as window slides

### Incremental Clustering

Stable, persistent cluster assignments (see `docs/incremental-clustering-design.md`):
- Each message assigned to exactly one cluster (or noise/fossil)
- Two-phase algorithm: centroid matching → HDBSCAN for new clusters
- Centroids evolve incrementally as new members join
- Noise reconsidered for N iterations before fossilizing
- Clusters persist across iterations with stable identity

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `k_samples` | 5 | Messages sampled per iteration |
| `active_pool_size` | 50 | Size of recency window |
| `model` | claude-haiku-4.5 | LLM for mind invocations |
| `min_cluster_size` | 3 | HDBSCAN threshold for new clusters |
| `centroid_match_threshold` | 0.3 | Max cosine distance for cluster matching |
| `noise_reconsider_iterations` | 20 | How long noise stays in candidate pool |

---

## Design Principles

### 1. Experimental Purity
Messages contain only content. Minds cannot see metadata.

### 2. Statefulness in Pool Only
Minds are stateless. The pool is the collective memory.

### 3. Non-destructive Exploration
Fork sessions to explore "what if" without losing state.

### 4. Observable Dynamics
Every action logged. Cluster evolution trackable.

---

## Session Format

### session.json

```json
{
  "iteration": 42,
  "config": {
    "model": "anthropic/claude-haiku-4.5",
    "k_samples": 5,
    "active_pool_size": 50
  }
}
```

### interventions.jsonl

```json
{"type": "inject", "content": {"text": "...", "vector_id": 5}, ...}
{"type": "run", "content": {"iterations": 10}, ...}
```

---

## Future Directions

**Clustering:**
- Cluster splitting when coherence drops
- Cluster shape metrics (aspect ratio, dimensionality)
- Cross-session cluster comparison

**Analysis:**
- Diversity metrics over time
- Injection impact analysis
- Cluster trajectory visualization

**Experiments:**
- Different models on same session
- Intervention strategies (diversity-weighted sampling)
- Message lineage tracking via sampled_ids
