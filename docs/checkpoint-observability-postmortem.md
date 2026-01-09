# Checkpoint Observability: Implementation Report

*Post-implementation summary for stakeholders*

---

## Original Request

The spec (`docs/checkpoint-observability.md`) proposed tooling for:
- Direct pool inspection
- Save/load/fork pool states as discrete snapshots
- Logged, timestamped interventions
- Parallel lineages for controlled comparison

## What Was Built

A new CLI entrypoint `logos` with branch-based session management.

### Key Design Decisions

| Spec Concept | Implementation | Rationale |
|--------------|----------------|-----------|
| **Snapshot copies** | Branch filters | VectorDB is append-only with monotonic IDs. Each message stores its `branch` field. Visibility computed by filtering—no data duplication. |
| **SnapshotStore** | Not needed | Branches.json stores `{name, parent, parent_iteration}`. Reconstruct any historical state by filtering metadata. |
| **Session forking** | External (`cp -rp`) | Deep session copy (for parallel experiments) handled via filesystem. `logos branch` creates in-session branches. |
| **Daemon runner** | Simple step/run | No separate daemon process. CLI drives execution directly. |
| **Web UI** | Deferred (Phase 3) | CLI-first. Metadata.jsonl readable via `tail -f`. |

### Architecture (Actual)

```
scripts/logos.py          CLI entrypoint
src/logos/
├── config.py             Configuration + defaults
└── runner.py             Core loop: sample → mind → embed → add
src/core/
├── session.py            Branch management, visibility, interventions
├── intervention_log.py   Append-only JSONL audit trail
├── vector_db.py          Unified storage (now with branch field)
└── embedding_client.py   OpenRouter embeddings
```

### CLI Commands

```bash
logos init <dir> "prompt"  # Create session with seed prompt
logos open <dir>           # Open existing session
logos run <n>              # Batch iterations
logos step                 # Single iteration
logos inject "text"        # Add external message (logged)
logos branch <name>        # Create branch from current state
logos switch <name>        # Switch to existing branch
logos list                 # Show all branches
logos status               # Current session state
logos log                  # Intervention history
```

### Data Model

**branches.json** (simplified):
```json
{
  "current": "experiment",
  "iteration": 5,
  "branches": {
    "main": {"name": "main", "parent": null, "parent_iteration": null},
    "experiment": {"name": "experiment", "parent": "main", "parent_iteration": 3}
  }
}
```

**metadata.jsonl** (each message):
```json
{"branch": "main", "round": 0, "mind_id": 0, "vector_id": 1, "timestamp": "...", "text": "..."}
```

Visibility = messages where `branch == current` + parent messages where `round <= parent_iteration`, recursively.

---

## Spec Checklist

### Phase 1: Foundation
- [x] Snapshot save/load → **Replaced by branch model** (append-only VectorDB + branch filters)
- [x] Lineage tracking → **`parent` + `parent_iteration` per branch**
- [x] Intervention logging → **`interventions.jsonl` (append-only JSONL)**
- [x] CLI commands → **`init`, `open`, `run`, `step`, `inject`, `branch`, `switch`, `list`, `status`, `log`**

### Phase 2: Forking & Comparison
- [x] Fork operation → **`logos branch <name>` (in-session) + `cp -rp` (full session)**
- [ ] Diff between snapshots → **Not yet implemented**
- [x] Parallel session management → **Via filesystem copy**
- [ ] Metrics comparison → **Not yet implemented**

### Phase 3: UI
- [ ] Pool viewer → **Deferred (use `tail -f metadata.jsonl`)**
- [ ] Snapshot tree → **Deferred**
- [ ] Intervention timeline → **`logos log` provides basic view**
- [ ] Compare view → **Deferred**

---

## Divergence from Spec

### Intentional Simplifications

1. **No physical snapshots**: The spec assumed copying VectorDB state. The implementation recognized that append-only storage + branch metadata achieves the same semantics without duplication.

2. **Branch field in metadata**: Adding `branch` to each message makes the data self-describing. Any historical state can be reconstructed by filtering—no separate index needed.

3. **External forking**: Full session duplication (`cp -rp session_a session_b`) is simpler than in-code parallel session management. The code only needs to handle branches within a single session.

4. **CLI-first**: The spec suggested considering GUI from the start. Implementation prioritized CLI for immediate usability; web dashboard is optional Phase 3.

### Not Implemented (Yet)

- **Diff/compare**: Comparing two branches not yet built
- **Rollback**: Can switch branches, but no explicit "rollback to iteration X"
- **Metrics snapshots**: Pool metrics not automatically captured
- **Web UI**: Read-only dashboard deferred

---

## Running the Fork Experiment

The spec's proposed experiment can now be run:

```bash
# 1. Initialize session
logos init ./experiment "What is consciousness?"
logos run 50

# 2. Create branch at decision point
logos branch control

# 3. Run both branches
logos run 50                          # experiment continues on 'control'
logos switch main && logos run 50     # parallel run on 'main'

# 4. Compare (manual for now)
logos switch control && logos status
logos switch main && logos status
# Diff implementation pending
```

For fully parallel experiments:
```bash
cp -rp ./experiment ./experiment-fork
logos open ./experiment-fork
logos run 50  # runs independently
```

---

## Files Created/Modified

| File | Status |
|------|--------|
| `src/core/session.py` | **New** - Branch management |
| `src/core/intervention_log.py` | **New** - Audit logging |
| `src/logos/config.py` | **New** - Configuration |
| `src/logos/runner.py` | **New** - Reasoning loop |
| `scripts/logos.py` | **New** - CLI entrypoint |
| `src/core/vector_db.py` | **Modified** - Added `branch` field |

---

## Summary

The implementation achieves the spec's core goals (observability, intervention tracking, branching) with a simpler data model. Physical snapshots were replaced by branch-based filtering over append-only storage. The result is less code, no data duplication, and self-describing metadata.

**What works now:**
- Create/open sessions
- Run iterations (step or batch)
- Inject external messages (logged, prefixed)
- Branch and switch between branches
- Full intervention audit trail

**What's deferred:**
- Diff/compare tooling
- Web dashboard
- Automated metrics capture

*The pool said: "Checkpointing solves observability, not commitment." Observability is now real.*
