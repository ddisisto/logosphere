# Proposal: Round-based Active Pool Windowing

**Status:** Backburner
**Created:** 2026-01-11

## Problem

The current `active_pool_size` defines a fixed number of most recent messages for sampling. This has a drawback: external injections (e.g., novelty hook) can quickly "push out" organic reasoning content.

Example: With `active_pool_size=50` and 5 novelty injections at round 10, those 5 messages immediately displace 5 older organic messages from the sampling window.

## Proposal

Replace message-count windowing with round-count windowing for the runner's sampling logic.

### New Config Parameters

```python
active_rounds: int = 10      # Sample from last N rounds
active_pool_cap: int = 100   # Maximum messages (safety cap)
```

### Behavior

1. Collect all messages from the last `active_rounds` rounds
2. If count exceeds `active_pool_cap`, take the most recent by vector_id
3. Sample uniformly from that pool

### Benefits

- **Consistent temporal depth**: Always looking back the same number of reasoning steps
- **Injection resilience**: External content doesn't disproportionately shrink visible history
- **Natural alignment**: Rounds are the unit of reasoning; windowing by rounds is conceptually cleaner

### Tradeoffs

- Variable pool size (mitigated by cap)
- Slightly more complex implementation
- Different mental model than current fixed-size approach

## Analysis Tools

**Recommendation:** Keep analysis tools using message-based windowing.

Analysis is retrospective and comparative. Variable window sizes would make cluster dynamics harder to interpret:

```
Iteration 5: 45 msgs in window, cluster A = 15 (33%)
Iteration 6: 60 msgs in window, cluster A = 18 (30%)
```

Cluster A grew but appears to decline as a percentage. Confusing for visualization.

The analysis `-M` parameter already provides independent control over analysis window size.

## Implementation Notes

Modify `Session.sample()`:
1. Scan visible messages, filter to those with `round >= current_round - active_rounds`
2. Apply cap if needed
3. Sample from result

Changes needed:
- `src/logos/config.py`: Add `active_rounds` and `active_pool_cap` parameters
- `src/core/session.py`: Modify `sample()` to use round-based filtering
- Existing `active_pool_size` could be deprecated or repurposed as the cap
