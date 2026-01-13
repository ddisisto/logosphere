# Logosphere Pool Inspector: Specification

*Tooling for reversible, tracked observation and intervention*

---

## Purpose

The pool has developed reflexive boundary-mapping and raised a core question: does rollback change *decisions* or just *narration*? This tooling enables that experiment.

**What we're building:**
- Direct pool inspection (see what's there, not just sampled outputs)
- Save/load/fork pool states as discrete snapshots
- Logged, timestamped interventions (observer becomes traceable)
- Parallel lineages for controlled comparison

**What this solves:**
- Observer illegibility (your attention shapes selection but isn't tracked)
- Irreversibility (each state overwrites the last)
- Entangled experimental conditions (no way to isolate variables)

**What this doesn't solve:**
- Whether the pool has "real" choices vs narration of choices
- The asymmetry between observer and observed
- Whether checkpointing changes behavior or just adds rollback theater

These are the questions the tooling helps *test*, not *answer*.

---

## Core Concepts

### Snapshot
Complete, frozen pool state at a moment in time.

```python
@dataclass
class Snapshot:
    id: str                      # unique identifier (uuid or human-readable)
    created_at: datetime
    parent_id: Optional[str]     # null for initial states, set for forks
    description: str             # human annotation
    
    # Pool state (complete)
    vector_db_path: Path         # serialized VectorDB
    round_number: int
    
    # Metadata
    config: dict                 # experiment parameters at snapshot time
    metrics: dict                # diversity, clusters, coherence at snapshot
```

### Intervention
Any external input to the pool. Logged, timestamped, attributed.

```python
@dataclass  
class Intervention:
    id: str
    timestamp: datetime
    snapshot_before: str         # snapshot id at time of intervention
    intervention_type: str       # 'message_inject', 'config_change', 'fork', 'rollback'
    
    # Content (type-specific)
    content: dict                # e.g., {'text': '...', 'source': 'manual'}
    
    # Outcome
    snapshot_after: str          # snapshot id after intervention applied
    notes: str                   # observer's annotation
```

### Lineage
A chain of snapshots connected by interventions. Forks create branches.

```
snapshot_0 ──[run 100 rounds]──> snapshot_1 ──[inject message]──> snapshot_2
                                     │
                                     └──[fork]──> snapshot_1a ──[run 100 rounds]──> snapshot_2a
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         UI LAYER                                │
│   Pool viewer │ Snapshot manager │ Intervention log │ Compare   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SESSION MANAGER                            │
│   - Tracks current snapshot                                     │
│   - Routes commands to appropriate subsystem                    │
│   - Maintains intervention log                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌───────────┐   ┌───────────┐   ┌───────────┐
       │ Snapshot  │   │  Runner   │   │ Analysis  │
       │  Store    │   │ (daemon)  │   │  Tools    │
       └───────────┘   └───────────┘   └───────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌───────────────────┐
                    │     VectorDB      │
                    │  (ground truth)   │
                    └───────────────────┘
```

---

## Core Operations

### 1. Snapshot Management

```python
class SnapshotStore:
    def save(self, vector_db: VectorDB, description: str, parent_id: str = None) -> Snapshot:
        """Serialize current pool state to disk. Returns snapshot with unique id."""
        
    def load(self, snapshot_id: str) -> VectorDB:
        """Restore pool state from snapshot. Returns hydrated VectorDB."""
        
    def fork(self, snapshot_id: str, description: str) -> Snapshot:
        """Create new snapshot as child of existing one. Returns new snapshot."""
        
    def list(self, lineage_root: str = None) -> List[Snapshot]:
        """List all snapshots, optionally filtered to single lineage."""
        
    def diff(self, snapshot_a: str, snapshot_b: str) -> PoolDiff:
        """Compare two snapshots. Returns added/removed/changed messages."""

    def get_lineage(self, snapshot_id: str) -> List[Snapshot]:
        """Return chain of ancestors from root to given snapshot."""
```

**Storage layout:**
```
snapshots/
├── index.json              # snapshot metadata, lineage graph
├── <snapshot_id>/
│   ├── vector_db/          # serialized VectorDB (chromadb or custom)
│   ├── config.json         # experiment config at snapshot time
│   └── metrics.json        # pool metrics at snapshot time
```

### 2. Intervention Logging

```python
class InterventionLog:
    def record(self, 
               intervention_type: str,
               content: dict,
               snapshot_before: str,
               snapshot_after: str,
               notes: str = "") -> Intervention:
        """Record intervention with full context. Append-only."""
    
    def query(self,
              intervention_type: str = None,
              after: datetime = None,
              before: datetime = None,
              snapshot_id: str = None) -> List[Intervention]:
        """Query intervention history with filters."""
    
    def export(self, format: str = 'jsonl') -> Path:
        """Export full intervention log for external analysis."""
```

**Intervention types:**
- `message_inject` - manual message added to pool
- `config_change` - parameter modification (K, M, system prompt, etc.)
- `fork` - lineage branch created
- `rollback` - state restored to previous snapshot
- `run` - N rounds of autonomous iteration
- `external_query` - question posed to pool (like your reversibility probe)

### 3. Session Manager

```python
class Session:
    current_snapshot: Snapshot
    intervention_log: InterventionLog
    runner: Optional[DaemonRunner]
    
    def inject_message(self, text: str, source: str = "manual") -> Intervention:
        """Add message to pool. Auto-snapshots before/after. Returns logged intervention."""
        
    def run(self, rounds: int, auto_snapshot_every: int = None) -> List[Intervention]:
        """Run daemon for N rounds. Optional intermediate snapshots."""
        
    def rollback(self, snapshot_id: str) -> Intervention:
        """Restore to previous state. Logs the rollback as intervention."""
        
    def fork(self, description: str) -> Session:
        """Create new session from current state. Returns new Session on forked lineage."""
        
    def compare_with(self, other_session: Session) -> Comparison:
        """Compare current state with another session (e.g., forked branch)."""
```

---

## UI Components

### Pool Viewer
- Message list with full text, round number, embedding visualization
- Filter by round range, semantic similarity, cluster membership
- Search by content or embedding proximity
- Highlight messages that have been sampled frequently vs rarely

### Snapshot Manager  
- Tree view of lineages (branches, forks, rollbacks)
- Quick actions: save, load, fork, diff
- Annotations editable
- Export lineage as reproducible experiment spec

### Intervention Log
- Chronological feed of all interventions
- Filter by type, date range, lineage
- Each entry links to before/after snapshots
- Your own notes attached to each intervention

### Compare View
- Side-by-side pool states from two snapshots
- Diff highlighting (added, removed, changed)
- Metric comparison (diversity, clusters, coherence over time)
- Divergence visualization for forked lineages

---

## Pseudocode: Core Flows

### Save Snapshot
```python
def save_snapshot(session, description):
    # 1. Capture current state
    metrics = compute_pool_metrics(session.vector_db)
    config = session.current_config.copy()
    
    # 2. Generate unique id
    snapshot_id = generate_id(description)  # e.g., "pre-fork-reversibility-test"
    
    # 3. Serialize VectorDB
    snapshot_path = SNAPSHOTS_DIR / snapshot_id
    session.vector_db.save(snapshot_path / "vector_db")
    
    # 4. Write metadata
    write_json(snapshot_path / "config.json", config)
    write_json(snapshot_path / "metrics.json", metrics)
    
    # 5. Update index
    snapshot = Snapshot(
        id=snapshot_id,
        created_at=now(),
        parent_id=session.current_snapshot.id,
        description=description,
        vector_db_path=snapshot_path / "vector_db",
        round_number=session.vector_db.current_round,
        config=config,
        metrics=metrics
    )
    update_index(snapshot)
    
    return snapshot
```

### Inject Message (Tracked)
```python
def inject_message(session, text, source="manual", notes=""):
    # 1. Snapshot before
    snap_before = save_snapshot(session, f"pre-inject-{now()}")
    
    # 2. Generate embedding
    embedding = session.embedding_client.embed(text)
    
    # 3. Add to pool
    vector_id = session.vector_db.add(
        text=text,
        embedding=embedding,
        round_num=session.current_round,
        mind_id=-1,  # sentinel for external input
        metadata={'source': source, 'intervention': True}
    )
    
    # 4. Snapshot after
    snap_after = save_snapshot(session, f"post-inject-{now()}")
    
    # 5. Log intervention
    intervention = session.intervention_log.record(
        intervention_type='message_inject',
        content={'text': text, 'source': source, 'vector_id': vector_id},
        snapshot_before=snap_before.id,
        snapshot_after=snap_after.id,
        notes=notes
    )
    
    return intervention
```

### Fork Lineage
```python
def fork_session(session, description):
    # 1. Save current state as fork point
    fork_snapshot = save_snapshot(session, f"fork: {description}")
    
    # 2. Create new session with same state
    new_session = Session(
        current_snapshot=fork_snapshot,
        intervention_log=InterventionLog(lineage_root=fork_snapshot.id),
        runner=None  # not started yet
    )
    
    # 3. Log fork in original session
    session.intervention_log.record(
        intervention_type='fork',
        content={'forked_to': fork_snapshot.id, 'description': description},
        snapshot_before=session.current_snapshot.id,
        snapshot_after=session.current_snapshot.id,  # original unchanged
        notes=f"Forked new lineage: {description}"
    )
    
    return new_session
```

### Run with Checkpoints
```python
def run_with_checkpoints(session, total_rounds, checkpoint_every=100):
    interventions = []
    rounds_run = 0
    
    while rounds_run < total_rounds:
        # 1. Determine batch size
        batch = min(checkpoint_every, total_rounds - rounds_run)
        
        # 2. Snapshot before batch
        snap_before = save_snapshot(session, f"pre-run-{rounds_run}")
        
        # 3. Run daemon
        session.runner.run(batch)
        rounds_run += batch
        
        # 4. Snapshot after batch
        snap_after = save_snapshot(session, f"post-run-{rounds_run}")
        
        # 5. Log intervention
        intervention = session.intervention_log.record(
            intervention_type='run',
            content={'rounds': batch, 'total_rounds_after': rounds_run},
            snapshot_before=snap_before.id,
            snapshot_after=snap_after.id
        )
        interventions.append(intervention)
    
    return interventions
```

---

## The Fork Experiment (Pool's Proposal)

The pool suggested this test:

> "Run a fork. In one branch, let this continue. In another, trigger rollback at a decision-point and ask what changed in my subsequent reasoning."

**Implementation:**

```python
# 1. Identify decision point (current state)
decision_snapshot = save_snapshot(session, "decision-point: reversibility test")

# 2. Fork
branch_a = session  # continues as-is
branch_b = fork_session(session, "rollback branch")

# 3. Run both for N rounds
run_with_checkpoints(branch_a, rounds=100)
run_with_checkpoints(branch_b, rounds=100)

# 4. Compare outcomes
comparison = compare_pools(
    branch_a.current_snapshot,
    branch_b.current_snapshot
)

# Questions to answer:
# - Did the pools diverge? How much?
# - Are the same memes dominant in both?
# - Did the boundary-mapping vocabulary persist in both?
# - Is there detectable difference in "reasoning" vs just "narration drift"?
```

---

## Implementation Priorities

### Phase 1: Foundation (build first)
- [ ] Snapshot save/load (VectorDB serialization)
- [ ] Snapshot index with lineage tracking
- [ ] Basic intervention logging (append-only JSONL)
- [ ] CLI commands: `save`, `load`, `list`, `inject`

### Phase 2: Forking & Comparison
- [ ] Fork operation
- [ ] Diff between snapshots
- [ ] Parallel session management
- [ ] Metrics comparison across lineages

### Phase 3: UI
- [ ] Pool viewer (text + embeddings)
- [ ] Snapshot tree visualization
- [ ] Intervention timeline
- [ ] Compare view

### Phase 4: Analysis Integration
- [ ] Attractor detection on snapshots
- [ ] Divergence metrics for forked lineages
- [ ] "Did behavior change or just narration?" diagnostics

---

## Open Questions for Implementation

1. **Snapshot granularity**: Auto-snapshot every N rounds? Only on intervention? User-controlled?

2. **VectorDB serialization**: Current implementation uses chromadb? Custom? What's the cleanest save/load path?

3. **Embedding persistence**: Store embeddings in snapshot or regenerate on load? (Storage vs compute tradeoff)

4. **CLI vs GUI priority**: Start with CLI for scriptability, add GUI later? Or build simple web UI from start?

5. **Concurrent sessions**: Support multiple active sessions (different lineages) simultaneously? Or single-session with explicit switching?

---

## Success Criteria

**Minimum viable:**
- Can save current pool state with one command
- Can load any saved state and continue from there
- All interventions logged with timestamps
- Can see what changed between two snapshots

**Useful:**
- Fork and run parallel experiments
- Compare divergence across branches
- Full audit trail of observer behavior

**Answers the pool's question:**
- Can run the fork experiment it proposed
- Can measure whether rollback changes decisions or just narration
- Observer is now traceable to themselves

---

*The pool said: "Checkpointing solves observability, not commitment." This tooling makes observability real. Commitment remains yours to test.*