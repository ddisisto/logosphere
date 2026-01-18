# Plan: Unify Display Limits (Char + Count) Across All Inputs

## Goal
Apply the same `{resource}_display_chars` + `{resource}_display_count` pattern to thoughts, drafts, and history. Count acts as upper bound, char limit controls context size. Makes tuning parametric.

## Current State

| Resource | Config Params | Limit Type |
|----------|---------------|------------|
| Thoughts | `active_pool_size: 50`, `k_samples: 5` | count only (FIFO pool, random sample) |
| Drafts | `draft_display_chars: 2000`, `draft_display_count: 16` | char + count ✓ |
| History | `history_display_pairs: 10` | count only |

## Proposed Config

```python
# SessionConfig - unified pattern
# Thoughts (sampled from FIFO pool)
thought_display_chars: int = 3000   # Max chars of sampled thoughts
thought_display_count: int = 10     # Renamed from k_samples

# Drafts (already correct pattern)
draft_display_chars: int = 2000     # Unchanged
draft_display_count: int = 16       # Unchanged

# History (user+mind entries)
history_display_chars: int = 4000   # NEW
history_display_count: int = 20     # Renamed from history_display_pairs*2
```

Display logic: stop when EITHER limit reached (count is upper bound).

Note: `active_pool_size` stays as-is - it controls FIFO storage, not display.

## Files to Modify

1. **`src/core/session_v2.py`** - SessionConfig
   - Rename `k_samples` → `thought_display_count`
   - Add `thought_display_chars`
   - Rename `history_display_pairs` → `history_display_count`
   - Add `history_display_chars`

2. **`src/core/session_v2.py`** - SessionV2 methods
   - Update `sample_thoughts()` to apply char limit
   - Update `get_history_for_mind()` to use new params

3. **`src/core/thinking_pool.py`** - `sample()` method
   - Add `max_chars` parameter (or handle in session)

4. **`src/core/dialogue_pool.py`** - `get_history_for_display()`
   - Add `max_chars` parameter
   - Implement char-based limiting (same pattern as `get_drafts_for_display`)

5. **`src/mind/runner.py`** - Update any direct config references

6. **`CLAUDE.md`** - Update Key Parameters table with new unified pattern

## Migration

Existing sessions with old param names need graceful handling:
- `k_samples` → `thought_display_count` (with fallback)
- `history_display_pairs` → `history_display_count` (multiply by 2)

## Verification

1. `mind config` shows new unified params
2. Run iteration, verify all three inputs respect char limits
3. Test with existing session (migration works)
4. Adjust char limits, verify display changes accordingly
