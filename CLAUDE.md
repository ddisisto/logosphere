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
│   │   ├── message_store.py   # Append-only message storage (user + mind)
│   │   ├── draft_store.py     # Append-only draft storage (0-indexed per message)
│   │   ├── session_v2.py      # Session management (thinking + messages + drafts)
│   │   ├── mind_v2.py         # YAML-based LLM invocation (v1.5 protocol)
│   │   ├── embedding_client.py # OpenRouter embedding API
│   │   ├── lock.py            # Session locking for TUI/CLI coordination
│   │   └── intervention_log.py # Append-only audit trail
│   ├── mind/
│   │   ├── runner.py          # Core loop: sample → mind → embed → cluster
│   │   ├── ops.py             # Shared operations for CLI and TUI
│   │   ├── events.py          # Event emitter for UI updates
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
│   ├── tui.py                 # TUI entry point
│   ├── logos.py               # Legacy CLI (v1)
│   └── extract_session.py     # Session extraction/forking utility
└── docs/
    ├── system_prompt_v1.5.md  # Current Mind protocol spec
    ├── dialogue-v2-design.md  # Dialogue data model design
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
mind drafts seen 0 2                   # Mark specific drafts as seen (0-indexed)
mind drafts show 0                     # Show full text of draft #0
mind accept                            # Accept latest draft
mind accept 2                          # Accept draft #2 (0-indexed)
mind history                           # Show conversation history
```

Notes:
- Cannot send a new message while awaiting response. Must accept a draft first.
- Cannot run iterations while idle. Must send a message first.
- Draft indices are 0-based and reset each message round.

### Configuration

```bash
mind config                            # Show current config
mind config --json                     # JSON output
mind config --set model=anthropic/claude-haiku-4.5
mind config --set thought_display_count=15
```

### Clustering

```bash
mind cluster status                    # Show cluster registry state
mind cluster show cluster_0            # Show members of a cluster
```

Clustering auto-initializes on first iteration - no bootstrap required.

### User Signal

```bash
mind signal                            # Show current signal
mind signal -a                         # Show all signal history
mind signal -p reviewing               # Set presence (a/r/e shortcuts)
mind signal -s "focusing on X"         # Set status text
mind signal -p e -s "deep work"        # Set both presence and status
```

Presence states: `absent` (a), `reviewing` (r), `engaged` (e). Default on init: `absent`.

---

## Core Concepts

### Dual-Pool Architecture

**Thinking Pool** (`thinking/`)
- Internal thoughts with embeddings
- Sampled randomly each iteration (not sequential)
- Clustered by semantic similarity
- FIFO rotation: oldest thoughts displaced when pool is full
- Minds see: text + age + cluster assignment

**Dialogue** (`messages.yaml` + `drafts.yaml`)
- Draft-based user ↔ mind communication
- States: idle (last message from mind) or drafting (last message from user)
- User sends message → mind produces drafts → user accepts one
- Messages: append-only, contains user messages + accepted mind responses
- Drafts: append-only, 0-indexed per message round, all drafts preserved
- State derived from last message role (no separate "awaiting" field)

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
- `messages.yaml` - User messages + accepted mind responses (append-only)
- `drafts.yaml` - All drafts, 0-indexed per message round (append-only)
- `clusters/` - Cluster registry and assignments
- `prompts/` - Raw LLM requests/responses (`{iter:06d}-req.txt`, `{iter:06d}-resp.txt`)
- `interventions.jsonl` - Audit log of all actions

Sessions are linear (no branching). Fork sessions by copying with `extract_session.py`.

### Mind Protocol (v1.5)

YAML-based input/output format. Block order: meta → thinking_pool → dialogue → drafts → orientation.

**Input (drafting state with history):**
```yaml
meta:
  self: mind_0
  iter: 247
  user_time: 2026-01-15T14:30:00+11:00
  limits:
    thoughts: {chars: 3000, count: 10}
    history: {chars: 4000, count: 20}
    drafts: {chars: 2000, count: 16}
  signal_state:
    consecutive_hard: 0
    threshold: 3

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

# Draft responses (most recent = last in list)
drafts:
  - |  # index: 1, age: 38, user_seen: true
    first draft response
  - |  # index: 2, age: 15, user_seen: false
    latest draft response

# Re-orientation after long context
orientation:
  iter: 247
  signal_state:
    consecutive_hard: 0
    threshold: 3
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

### Display Limits (Unified Pattern)

All inputs to the mind use a consistent `{resource}_display_chars` + `{resource}_display_count` pattern. Count acts as upper bound, char limit controls context size.

| Resource | Char Limit | Count Limit | Effect |
|----------|------------|-------------|--------|
| Thoughts | `thought_display_chars: 3000` | `thought_display_count: 10` | Sampled thoughts shown to mind |
| Drafts | `draft_display_chars: 2000` | `draft_display_count: 16` | Recent drafts shown to mind |
| History | `history_display_chars: 4000` | `history_display_count: 20` | Conversation history shown to mind |

### Other Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `active_pool_size` | 50 | Size of thinking pool (FIFO storage, not display) |
| `model` | claude-haiku-4.5 | LLM for mind invocations |
| `token_limit` | 4000 | Max tokens for LLM response |
| `min_cluster_size` | 3 | HDBSCAN threshold for new clusters |
| `centroid_match_threshold` | 0.3 | Max cosine distance for cluster matching |
| `hard_signal_threshold` | 3 | Consecutive hard signals before forced stop |

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
  thought_display_chars: 3000
  thought_display_count: 10
  active_pool_size: 50
  draft_display_chars: 2000
  draft_display_count: 16
  history_display_chars: 4000
  history_display_count: 20
  model: anthropic/claude-haiku-4.5
  token_limit: 4000
  embedding_model: openai/text-embedding-3-small
  embedding_dim: 1536
  min_cluster_size: 3
  centroid_match_threshold: 0.3
  hard_signal_threshold: 3
```

### messages.yaml

Append-only list of user messages and accepted mind responses:

```yaml
- role: user
  iter: 5
  time: "2026-01-15T10:00:00+00:00"
  text: |
    Hello, how are you?

- role: mind
  iter: 12
  time: "2026-01-15T10:05:00+00:00"
  text: |
    I'm doing well, thank you.
  draft_index: 2
  user_message_iter: 5

- role: user
  iter: 15
  time: "2026-01-15T10:10:00+00:00"
  text: |
    Great to hear!
```

State derivation: if last message is `role: user` → DRAFTING, otherwise → IDLE.

### drafts.yaml

Append-only list of all drafts, 0-indexed per message round:

```yaml
- user_message_iter: 5
  index: 0
  iter: 7
  time: "2026-01-15T10:01:00+00:00"
  text: |
    First draft...
  seen: false

- user_message_iter: 5
  index: 1
  iter: 9
  time: "2026-01-15T10:02:00+00:00"
  text: "+1"
  seen: true

- user_message_iter: 5
  index: 2
  iter: 12
  time: "2026-01-15T10:05:00+00:00"
  text: |
    I'm doing well, thank you.
  seen: true
```

Draft `index` is 0-based, resets each message round. Query by `user_message_iter` for all drafts for a message.

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
