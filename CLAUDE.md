# CLAUDE.md - Logosphere Architecture (v2)

## Experimental Boundary

**INSIDE THE EXPERIMENT:**
- Thoughts in the thinking pool
- Dialogue state (awaiting message, drafts, history)
- Mind inputs (sampled thoughts + dialogue state + system prompt)
- Mind outputs (thoughts + draft)

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
│   │   ├── dialogue_pool.py   # Draft-based dialogue (awaiting/drafts/history)
│   │   ├── session_v2.py      # Dual-pool session management
│   │   ├── mind_v2.py         # YAML-based LLM invocation (v1.2 protocol)
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
    ├── system_prompt_v1.2.md  # Current Mind protocol spec
    ├── draft-dialogue-design.md # Draft dialogue design doc
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
mind run                               # Run until stop (observe mode, max 100)
mind run -b                            # Background: drafts unseen, stop on hard signal only
mind run 10                            # Run exactly 10 iterations
mind run -b 10                         # Background: 10 iterations, drafts unseen
mind step                              # Single iteration
mind step --debug                      # Dump full LLM request/response
```

**Modes:**
- Observe (default): drafts marked seen, stops on each draft
- Background (`-b`): drafts unseen, stops on hard signal only (3+ consecutive no-drafts, or true silence)

### Dialogue

```bash
mind message "hello"                   # Send message to mind (starts drafting)
mind message -f prompt.md              # Send from file
cat prompt.md | mind message           # Send via pipe
mind drafts                            # Show current drafts (newest first)
mind drafts seen                       # Mark all drafts as seen
mind drafts seen 1 3                   # Mark specific drafts as seen
mind drafts archive                    # List all archived exchanges
mind drafts archive exc_42_000         # Show all drafts for an exchange
mind accept                            # Accept latest draft
mind accept 2                          # Accept specific draft
mind history                           # Show conversation history
```

Notes:
- Cannot send a new message while awaiting response. Must accept a draft first.
- Cannot run iterations while idle. Must send a message first.

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

**Dialogue Pool** (`dialogue/`)
- Draft-based user ↔ mind communication
- States: idle (history only) or drafting (awaiting + drafts)
- User sends message → mind produces drafts → user accepts one
- Accepted exchanges form conversation history (unlimited storage)
- All drafts stored (unlimited), but mind sees display-limited subset
- All drafts archived to `draft_archive.jsonl` when exchange completes
- Active drafts cleared after archiving; archive is append-only forever

**Signal Channel:**
- Draft buffer serves as bidirectional communication channel
- Hard signal: no draft output = demands user attention
- Soft signal: `+1` draft = endorses latest, still iterating
- True silence (no draft, no thoughts) = immediate stop signal

### Dialogue Flow

```
1. User sends message       → state becomes DRAFTING, iterations enabled
2. Mind iterations produce drafts (0 or more per iteration)
3. User marks drafts as seen (optional)
4. User accepts one draft   → state becomes IDLE, iterations blocked
5. Exchange added to history
6. Repeat from step 1
```

Strict mode: iterations only run during drafting state. When idle, user must send a message to continue.

### Session

A session is a directory containing:
- `session.yaml` - Iteration counter and config
- `thinking/` - Thought embeddings and pool state
- `dialogue/` - Dialogue pool state (awaiting/drafts/history)
- `clusters/` - Cluster registry and assignments
- `interventions.jsonl` - Audit log of all actions

Sessions are linear (no branching). Fork sessions by copying with `extract_session.py`.

### Mind Protocol (v1.2)

YAML-based input/output format.

**Input (drafting state with history):**
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

dialogue:
  history:
    - from: user
      age: 200
      text: |
        previous user message
    - from: self
      age: 195
      text: |
        accepted response
  awaiting:
    age: 42
    text: |
      user's message awaiting response
  drafts:
    - |  # age: 38, user_seen: true
      first draft response
    - |  # age: 15, user_seen: false
      latest draft response
```

**Output:**
```yaml
thoughts:
  - a brief thought
  - |
    multi-line thought using
    YAML block format

draft: |
  response to user's message
  (complete and self-contained)
```

Optional outputs: `thoughts: []`, no `draft:`, or `skip: true` for silence.

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
| `draft_display_chars` | 2000 | Max chars of drafts to show mind |
| `draft_display_count` | 16 | Max number of drafts to show mind |
| `history_display_pairs` | 10 | Conversation pairs to show mind |
| `model` | claude-haiku-4.5 | LLM for mind invocations |
| `token_limit` | 4000 | Max tokens for LLM response |
| `min_cluster_size` | 3 | HDBSCAN threshold for new clusters |
| `centroid_match_threshold` | 0.3 | Max cosine distance for cluster matching |

---

## Design Principles

### 1. Experimental Purity
Thoughts/drafts contain only content. Minds see relative age, not absolute time.

### 2. Dual-Pool Separation
Thinking (internal, clustered, sampled) vs dialogue (external, draft-based, sequential).

### 3. Draft-Based Refinement
Mind can refine responses over multiple iterations before user accepts.

### 4. Statefulness in Pools Only
Minds are stateless. The pools are the collective memory.

### 5. Non-destructive Exploration
Fork sessions to explore "what if" without losing state.

### 6. Observable Dynamics
Every action logged. Cluster evolution trackable.

---

## Session Format

### session.yaml

```yaml
iteration: 247
config:
  k_samples: 5
  active_pool_size: 50
  draft_display_chars: 2000
  draft_display_count: 16
  history_display_pairs: 10
  model: anthropic/claude-haiku-4.5
  token_limit: 4000
  embedding_model: openai/text-embedding-3-small
  embedding_dim: 1536
  min_cluster_size: 3
  centroid_match_threshold: 0.3
```

### dialogue/pool.yaml

```yaml
awaiting:
  iter: 245
  time: 2026-01-15T14:30:00+00:00
  text: |
    user's message
drafts:
  - iter: 246
    time: 2026-01-15T14:31:00+00:00
    text: |
      draft response
    seen: true
history:
  - role: user
    iter: 100
    time: 2026-01-15T10:00:00+00:00
    text: |
      earlier message
  - role: mind
    iter: 105
    time: 2026-01-15T10:05:00+00:00
    text: |
      earlier response
    accepted_draft_index: 3
    draft_archive_id: exc_100_000
```

### dialogue/draft_archive.jsonl

Append-only archive of all drafts from completed exchanges. Each line is a JSON object:

```json
{"exchange_id": "exc_100_000", "draft_index": 1, "iter_created": 102, "time_created": "2026-01-15T10:02:00+00:00", "text": "first draft...", "user_seen": true, "accepted": false, "accepted_by_exchange": null}
{"exchange_id": "exc_100_000", "draft_index": 2, "iter_created": 103, "time_created": "2026-01-15T10:03:00+00:00", "text": "second draft...", "user_seen": true, "accepted": false, "accepted_by_exchange": null}
{"exchange_id": "exc_100_000", "draft_index": 3, "iter_created": 105, "time_created": "2026-01-15T10:05:00+00:00", "text": "accepted response", "user_seen": true, "accepted": true, "accepted_by_exchange": "exc_100_000"}
```

Exchange IDs follow the format `exc_{awaiting_iter}_{sequence:03d}` where the sequence handles rare cases of multiple exchanges at the same iteration.

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

**Clustering:**
- Cluster splitting when coherence drops
- Cluster shape metrics (aspect ratio, dimensionality)
- Cross-session cluster comparison

**Analysis:**
- Diversity metrics over time
- Draft evolution analysis
- Cluster trajectory visualization

**Experiments:**
- Different models on same session
- Multi-mind sessions
- Thought lineage tracking

**Dialogue enhancements:**
- Threading (branching conversations)
- Draft annotations (confidence, "still working")
- Auto-accept after N iterations without new draft
