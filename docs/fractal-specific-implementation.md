# Hierarchical Pool Architecture (Fractal Spawning) - Implementation Plan

## Summary

Implement cluster-spawned child instances per `docs/fractal.md`. When a cluster is mentioned in parent thoughts (`[N]` syntax), spawn a child instance running identical protocol. Children develop specialized depth; parent sees child drafts via `cluster_reports`.

## Key Design Decisions (from RFC)

- **Reference Schema**: `[N]` single, `[N, M, K]` multi-cluster (regex: `\[(\d+(?:,\s*\d+)*)\]`)
- **Global Cluster IDs**: Single ID space across hierarchy (existing guarantee)
- **Visibility Mode**: ≤16 char = signal (stripped), ≥128 char = substantive (current state)
- **Cycle Allocation**: Derived from ref volume, not explicit params

## Design Principle: Parametric + Use Existing

Derive state from existing structures rather than adding explicit tracking:
- **Dormancy**: ref_count == 0 in active window → dormant (no threshold param)
- **Cycle rate**: proportional to refs in context, capped at 50% (no rate param)
- **Child status**: derived from refs, not stored state
- **Fairness**: proportional weighting between siblings (no ceiling param)

## Phased Implementation

### Phase 1: Data Model & Storage (Foundation)

**New files:**
- `src/core/child_instance.py` - ChildInstance class (minimal state)
- `src/core/mention_parser.py` - Parse `[N]` references from text

**Storage structure:**
```
session/
└── children/
    └── cluster_{N}/         # Per-child directory (existence = spawned)
        ├── thinking/        # Child's isolated pool
        └── drafts.yaml      # Child's draft buffer
```

Note: No registry, no ledger, no child.yaml - directory existence = spawned, status derived from live ref counts.

**Modify `session_v2.py`:**
- Add `get_children()` method (scan children/ directory)
- No new config params - derive from existing

### Phase 2: Spawn Mechanism

**In `child_instance.py`:**
- `ChildInstance.create(cluster_id, cluster_mgr, parent_pool)` - Create with bootstrap
- Bootstrap = cluster's current thoughts only (not full parent pool)
- Child gets own ThinkingPool and DraftStore

**In `runner.py`:**
- After parsing output, call `parse_mentions()` on thoughts
- Spawn child on first mention if cluster exists and child dir doesn't

### Phase 3: Cycle Allocation

**Derived scheduling (no new params):**
```python
# Count refs in current context
context_size = active_pool_size + draft_count + history_display_count
ref_ratio = total_child_refs_in_context / context_size
child_probability = min(0.5, ref_ratio)  # Cap at 50%

# Weight between children by individual ref counts (sqrt for diminishing returns)
child_weights = {cid: sqrt(ref_count) for cid, ref_count in child_refs.items()}
```

**Modify `runner.py`:**
- Add `_compute_allocation()` method (inline, no separate scheduler module)
- Add `_step_child(cluster_id)` method for child iterations

### Phase 4: Child Runner

**Add to `runner.py`:**
- `ChildRunner` class using same protocol but:
  - Isolated thinking pool (child's own)
  - Isolated draft buffer (child's own)
  - NO dialogue section (children don't receive user messages)
  - **Continuous flow**: Fresh parent cluster members included in child context each iteration
  - Parent thoughts mentioning cluster also flow into context

### Phase 5: Mind Input Extension (cluster_reports)

**Modify `mind_v2.py`:**
- Add `format_cluster_reports()` function
- Add `cluster_reports` parameter to `format_input()`

**Format:**
```yaml
cluster_reports:
  - cluster_id: 7
    presence: active
    visibility_mode: current
    top_draft: |
      child's synthesis...
    ref_count: 14
```

**Visibility logic:**
- Top draft ≤16 char → show reference-time snapshot
- Top draft ≥128 char → show current state
- Signals stripped from parent view entirely

### Phase 6: Dormancy & Lifecycle

**Derived from ref_count (no explicit tracking):**
- Dormant = ref_count == 0 in current context (automatic)
- Reactivation = any `[N]` ref appears (automatic)
- No explicit dormancy/reactivation methods needed - allocation naturally gives 0 cycles to unreferenced children

### Phase 7: Protocol Update

**New file: `docs/system_prompt_v1.6.md`**
- Document cluster reference syntax
- Document cluster_reports section
- Explain bidirectional visibility mechanics

### Phase 8: CLI & TUI

**CLI commands:**
```bash
mind children                    # List all children
mind children cluster_7          # Show child status
mind children cluster_7 drafts   # Show child's drafts
mind hierarchy stats             # Allocation statistics
```

**TUI panel:** `src/tui/panels/hierarchy_panel.py`

## Critical Files to Modify

| File | Changes |
|------|---------|
| `src/core/session_v2.py` | Add `get_children()` method |
| `src/mind/runner.py` | Mention parsing, allocation logic, `_step_child()` |
| `src/core/mind_v2.py` | Add cluster_reports section to format_input() |
| `scripts/mind.py` | Add `children` command |

## New Files

| File | Purpose |
|------|---------|
| `src/core/child_instance.py` | ChildInstance class with ThinkingPool + DraftStore |
| `src/core/mention_parser.py` | Parse `[N]` references from text |
| `docs/system_prompt_v1.6.md` | Protocol update with hierarchy docs |
| `src/tui/panels/hierarchy_panel.py` | TUI panel for child status |

## Configuration Parameters

**No new config params.** All derived from existing:

| Derived From | Used For |
|--------------|----------|
| `active_pool_size` + display counts | Context size for ref ratio |
| ref_count in context | Cycle allocation probability |
| ref_count == 0 | Dormancy detection |
| Draft text length | Visibility mode (≤16 signal, ≥128 substantive) |

**Hardcoded constants (can promote to config if tuning needed):**
- Child cycle cap: 50% of iterations max
- Mention scaling: sqrt (diminishing returns)
- Visibility thresholds: 16/128 chars (matches existing signal protocol)

## Verification

1. **Unit tests:** mention parsing, spawn logic, scheduler fairness
2. **Integration:** spawn on mention, cycle allocation, dormancy transitions
3. **End-to-end:** Run session, observe natural spawning, verify cluster_reports appear

## Migration

Existing sessions work unchanged - hierarchy is additive:
- No `children/` directory = no hierarchy active
- Global cluster IDs already unique (no migration needed)

**Current active session**: Can proceed without migration. Hierarchy features activate on first `[N]` mention that triggers spawn. Existing clusters, thoughts, drafts unaffected.

## Dependencies

No new external dependencies. Uses existing numpy, yaml, hdbscan.

---

## Resolved Decisions

| Question | Decision |
|----------|----------|
| Bootstrap | **Continuous flow** - new parent cluster members flow into child context |
| Audit ledger | **Skip** - count refs live each iteration, no mentions.jsonl |
| Depth | **Arbitrary** - full fractal from start, children can spawn grandchildren |
| Ref scope | **Full context** - thinking_pool + drafts + history |

## Remaining Assumptions

1. **Cluster must exist before spawn**: `[N]` references existing cluster only. Non-existent cluster refs are ignored.

2. **Child drafts isolated from CLI**: User sees child drafts via TUI panel and `cluster_reports`, not `mind drafts`.

3. **Single user model**: Children share parent's dialogue context (same awaiting message).

4. **Orphaned children**: If parent cluster dissolves, child continues with frozen bootstrap + continuous flow from any new cluster members (cluster ID persists even if temporarily empty).

## Simplifications Applied

1. ~~mentions.jsonl~~ → Count refs live (no separate file)
2. ~~child_registry.yaml~~ → Directory existence = spawned
3. ~~child.yaml status~~ → Derive from ref_count
4. ~~scheduler.py~~ → Inline in runner.py
5. ~~explicit config params~~ → Derive from existing

## Testing Considerations

- Trigger spawning: Create cluster first (min 3 thoughts), then reference `[N]`
- Cycle fairness: Mock random selection, verify weights
- TUI integration: Hierarchy panel subscribes to same events as draft view
