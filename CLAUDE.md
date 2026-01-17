# CLAUDE.md - Logosphere Architecture (v2)

## Experimental Boundary

**INSIDE THE EXPERIMENT:**
- Thoughts in the thinking pool
- Messages in the message pool
- Mind inputs (sampled thoughts + messages + system prompt)
- Mind outputs (thoughts + messages)

**OUTSIDE THE EXPERIMENT:**
- Runner (sampler, parser, embedder)
- API calls to LLM
- Analysis tools

**Critical:** Minds see only relative age (iterations since creation), not absolute timestamps or iteration numbers. Cluster IDs and sizes are visible as metadata on sampled thoughts.

---

## Current Structure

```
logosphere/
├── src/
│   ├── core/
│   │   ├── thinking_pool.py   # Embedded thoughts with FIFO rotation
│   │   ├── message_pool.py    # Direct communication (per-source FIFO)
│   │   ├── session_v2.py      # Dual-pool session management
│   │   ├── mind_v2.py         # YAML-based LLM invocation
│   │   ├── embedding_client.py # OpenRouter embedding API
│   │   └── intervention_log.py # Append-only audit trail
│   ├── mind/
│   │   ├── runner.py          # Core loop: sample → mind → embed → cluster
│   │   └── config.py          # Runtime configuration
│   ├── logos/
│   │   ├── clustering/        # Incremental clustering package
│   │   │   ├── models.py      # ClusterState, AssignmentTable
│   │   │   ├── algorithm.py   # Process iteration
│   │   │   └── manager.py     # ClusterManager persistence
│   │   └── analyze.py         # Legacy sliding window analysis
│   ├── analysis/              # Standalone analysis tools
│   ├── exchange/              # Auditor hooks (experimental)
│   └── tui/                   # Chat TUI interface
├── scripts/
│   ├── mind.py                # Main CLI (v2)
│   ├── logos.py               # Legacy CLI (v1)
│   └── extract_session.py     # Session extraction/forking utility
└── docs/
    ├── system_prompt_v1.1.md  # Current Mind protocol spec
    ├── draft-dialogue-design.md # Proposed draft-based dialogue
    └── ...                    # Other design docs
```

---

## Development Setup

```bash
uv sync                      # Install core dependencies
uv sync --extra analysis     # Add analysis tools (hdbscan, numpy)
```

Run mind: `python scripts/mind.py --help`
Run tests: `pytest`

---

## Mind CLI (v2)

The main interface for running dual-pool reasoning sessions.

### Session Management

```bash
mind init ./session "initial prompt"   # Create session with seed message
mind init ./session                    # Create empty session
mind open ./session                    # Open existing session
mind status                            # Show current state
```

### Running Iterations

```bash
mind run                               # Run until message emitted (default)
mind run 10                            # Run exactly 10 iterations
mind run --max 50                      # Safety limit for run-until-message
mind step                              # Single iteration
mind step --debug                      # Dump full LLM request/response
```

### Messaging

```bash
mind message "hello"                   # Send message to mind
mind message -f prompt.md              # Send from file
cat prompt.md | mind message           # Send via pipe
mind message --to mind_1 "hello"       # Send to specific mind (default: mind_0)
```

### Configuration

```bash
mind config                            # Show current config
mind config --json                     # JSON output
mind config --set model=anthropic/claude-haiku-4.5
mind config --set k_samples=10
```

### Clustering

```bash
mind cluster status                    # Show cluster registry state
mind cluster show cluster_0            # Show members of a cluster
```

Clustering auto-initializes on first iteration - no bootstrap required.

---

## Core Concepts

### Dual-Pool Architecture

**Thinking Pool** (`thinking/`)
- Internal thoughts with embeddings
- Sampled randomly each iteration (not sequential)
- Clustered by semantic similarity
- FIFO rotation: oldest thoughts displaced when pool is full
- Minds see: text + age + cluster assignment

**Message Pool** (`messages/`)
- Direct user ↔ mind communication
- Per-source FIFO buffers (user's messages separate from mind's)
- Messages include: source, recipient, age, timestamp
- No embeddings or clustering

### Session

A session is a directory containing:
- `session.yaml` - Iteration counter and config
- `thinking/` - Thought embeddings and pool state
- `messages/` - Message pool state
- `clusters/` - Cluster registry and assignments
- `interventions.jsonl` - Audit log of all actions

Sessions are linear (no branching). Fork sessions by copying with `extract_session.py`.

### Mind Protocol (v1.1)

YAML-based input/output format.

**Input:**
```yaml
meta:
  self: mind_0
  iter: 247
  user_time: 2026-01-15T14:30:00+11:00

thinking_pool:
  - |  # age: 50, cluster: {id: 3, size: 8}
    sampled thought with cluster context
  - |  # age: 12, cluster: {~}
    noise thought (no cluster yet)

message_pool:
  - source: user
    to: mind_0
    age: 42
    time: 2026-01-15T12:31:19+11:00
    text: |
      user's message
```

**Output:**
```yaml
thoughts:
  - a brief thought
  - |
    multi-line thought using
    YAML block format

messages:
  - to: user
    text: |
      response to user
```

Optional outputs: `thoughts: []`, `messages: []`, or `skip: true` for silence.

### Incremental Clustering

Stable, persistent cluster assignments (see `docs/incremental-clustering-design.md`):
- Each thought assigned to exactly one cluster (or noise `~`)
- Two-phase algorithm: centroid matching → HDBSCAN for new clusters
- Centroids evolve incrementally as new members join
- Noise stays in active pool, may cluster later
- Clusters persist across iterations with stable identity

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `k_samples` | 5 | Thoughts sampled per iteration |
| `active_pool_size` | 50 | Size of thinking pool |
| `message_buffer_per_source` | 10 | Messages retained per source |
| `model` | claude-haiku-4.5 | LLM for mind invocations |
| `token_limit` | 4000 | Max tokens for LLM response |
| `min_cluster_size` | 3 | HDBSCAN threshold for new clusters |
| `centroid_match_threshold` | 0.3 | Max cosine distance for cluster matching |

---

## Design Principles

### 1. Experimental Purity
Thoughts/messages contain only content. Minds see relative age, not absolute time.

### 2. Dual-Pool Separation
Thinking (internal, clustered, sampled) vs messaging (external, direct, complete).

### 3. Statefulness in Pools Only
Minds are stateless. The pools are the collective memory.

### 4. Non-destructive Exploration
Fork sessions to explore "what if" without losing state.

### 5. Observable Dynamics
Every action logged. Cluster evolution trackable.

---

## Session Format

### session.yaml

```yaml
iteration: 247
config:
  k_samples: 5
  active_pool_size: 50
  message_buffer_per_source: 10
  model: anthropic/claude-haiku-4.5
  token_limit: 4000
  embedding_model: openai/text-embedding-3-small
  embedding_dim: 1536
  min_cluster_size: 3
  centroid_match_threshold: 0.3
```

### interventions.jsonl

```json
{"type": "message", "content": {"source": "user", "text": "..."}, ...}
{"type": "run", "content": {"iterations": 10}, ...}
```

---

## Maintaining This Document

Update CLAUDE.md when:
- CLI commands or flags change
- New core concepts are introduced (pools, protocols, etc.)
- Session format or directory structure changes
- Key parameters are added/removed/renamed
- Design principles evolve

Design docs should include "Update CLAUDE.md" as a final implementation step.

---

## Future Directions

**Draft-Based Dialogue** (see `docs/draft-dialogue-design.md`):
- Mind produces draft responses, user accepts one
- Asynchronous refinement loop
- Non-accepted drafts pruned from history

**Clustering:**
- Cluster splitting when coherence drops
- Cluster shape metrics (aspect ratio, dimensionality)
- Cross-session cluster comparison

**Analysis:**
- Diversity metrics over time
- Message impact analysis
- Cluster trajectory visualization

**Experiments:**
- Different models on same session
- Multi-mind sessions
- Message lineage tracking
