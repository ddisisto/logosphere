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
│   │   ├── session.py         # Branch management, visibility
│   │   ├── intervention_log.py # Append-only audit trail
│   │   ├── embedding_client.py # OpenRouter embedding API
│   │   └── mind.py            # LLM invocation and parsing
│   └── logos/
│       ├── config.py          # Configuration defaults
│       ├── runner.py          # Core loop: sample → mind → embed → add
│       └── analyze.py         # Cluster timeline analysis
├── scripts/
│   └── logos.py               # Main CLI
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

### Branching

```bash
logos branch experiment                 # Create branch from current state
logos branch experiment -v 42           # Branch from specific vector_id
logos switch main                       # Switch branch
logos list                              # Show all branches
```

### Configuration

```bash
logos config                            # Show current branch config
logos config --set model=anthropic/claude-haiku-4.5
logos config --set k_samples=10
```

### Analysis

```bash
logos analyze                           # Cluster timeline (swimlane)
logos analyze --json                    # JSON output
logos analyze --min-cluster-size 2      # Tune HDBSCAN
logos analyze --centroid-threshold 0.3  # Tune cluster matching
```

### Intervention Log

```bash
logos log                               # Show intervention history
```

---

## Core Concepts

### Session

A session is a directory containing:
- `branches.json` - Branch metadata and per-branch iteration counters
- `vector_db/` - Embeddings and message metadata
- `interventions.jsonl` - Audit log of all actions

### Branches

Branches provide non-destructive exploration:
- Each branch has its own iteration counter (no gaps)
- New branches inherit parent's iteration at branch point
- Visibility is computed by filtering, not copying data
- Switch freely between branches

### VectorDB

Append-only message storage:
- Each message has: text, embedding, round, branch, mind_id
- Active pool = tail N messages visible to current branch
- Sampling is uniform random from active pool

### Mind

Stateless LLM invocation:
- Input: system prompt + K sampled messages
- Output: thinking (private) + transmitted messages (public)
- Parsing: `---` separates thoughts; first block is private

### Analysis

Cluster timeline shows semantic evolution:
- HDBSCAN clustering at each iteration
- Centroid matching tracks cluster identity across time
- ASCII swimlane visualization or JSON export

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `k_samples` | 5 | Messages sampled per iteration |
| `active_pool_size` | 50 | Size of recency window |
| `model` | claude-haiku-4.5 | LLM for mind invocations |
| `min_cluster_size` | 3 | HDBSCAN clustering threshold |

---

## Design Principles

### 1. Experimental Purity
Messages contain only content. Minds cannot see metadata.

### 2. Statefulness in Pool Only
Minds are stateless. The pool is the collective memory.

### 3. Non-destructive Exploration
Branches let you explore "what if" without losing state.

### 4. Observable Dynamics
Every action logged. Cluster evolution trackable.

---

## Session Format

### branches.json

```json
{
  "current": "main",
  "branches": {
    "main": {
      "name": "main",
      "parent": null,
      "parent_iteration": null,
      "iteration": 42,
      "config": { ... }
    },
    "experiment": {
      "name": "experiment",
      "parent": "main",
      "parent_iteration": 20,
      "iteration": 35,
      "config": { ... }
    }
  }
}
```

### interventions.jsonl

```json
{"type": "inject", "content": {"text": "...", "vector_id": 5}, ...}
{"type": "branch", "content": {"from_branch": "main", "to_branch": "exp"}, ...}
{"type": "run", "content": {"iterations": 10, "branch": "main"}, ...}
```

---

## Future Directions

**Analysis:**
- Diversity metrics over time
- Cross-branch comparison
- Injection impact analysis

**Experiments:**
- Different models on same session
- Intervention strategies (diversity-weighted sampling)
- Message lineage tracking via sampled_ids
