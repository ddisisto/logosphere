# Implementation: Structured Exchange Protocol

**Approach:** New parallel modules, no legacy changes until validated.

## Phase 1: Auditor Hook

**Goal:** Periodic summarization that enters the pool.

### Task 1.1: Create `hooks/auditor.py`
```python
AUDIT_EVERY = 20  # rounds
AUDITOR_MODEL = "anthropic/claude-sonnet-4"
AUDITOR_PROMPT = """You are an Auditor..."""

def hook(session, iteration, runner):
    if iteration % AUDIT_EVERY != 0 or iteration == 0:
        return
    # Sample broadly from pool
    # Invoke auditor LLM
    # Inject summary with [AUDIT] prefix
```

### Task 1.2: Auditor LLM invocation
- New function in `src/exchange/auditor.py`
- Takes pool sample + auditor prompt
- Returns summary text
- Separate from Mind - different model/prompt possible

### Task 1.3: Test independently
- Create test session
- Run with auditor hook enabled
- Verify summaries appear in pool

---

## Phase 2: Self-Reports

**Goal:** Inject role descriptions so Pool sees all prompts.

### Task 2.1: Create `hooks/self_report.py`
```python
REPORT_EVERY = 20  # same as audit, or different

def hook(session, iteration, runner):
    if iteration % REPORT_EVERY != 0:
        return
    # Inject [AUDITOR ROLE] with auditor's system prompt
    # Inject [OBSERVER ROLE] with observer's stated role
```

### Task 2.2: Config for role descriptions
- `auditor_prompt` in branch config (or file)
- `observer_role` in branch config (or file)
- Hooks read from config

---

## Phase 3: Observer Response

**Goal:** Human can respond to audits.

### Task 3.1: Manual approach first
- After audit round, user runs `logos inject "[OBSERVER] ..."`
- No automation needed initially

### Task 3.2: Later: `logos chat` mode
- Interactive loop: run N rounds → show audit → prompt for response → inject → continue
- New script: `scripts/logos_chat.py`

---

## Phase 4: Restart Detection

**Goal:** Detect when pool is signaling for restart.

### Task 4.1: Cluster-based detection
- Check if "restart" theme dominates (via existing analysis)
- Or simpler: grep for RESTART in recent messages

### Task 4.2: Hook or post-run check
- `hooks/restart_detector.py` - checks after each round
- Raises exception or returns signal to halt

### Task 4.3: Observer notification
- Print warning, pause for input
- Or automatic branch per config

---

## Phase 5: Cluster Analysis Feed

**Goal:** Pool/Auditor see embedding space dynamics.

### Task 5.1: Cluster summary injection
- After analysis, inject cluster info as message
- `[CLUSTERS] 3 clusters: A (45%), B (30%), C (15%) - themes: ...`

### Task 5.2: Real-time during run
- Optional: run cluster analysis every M rounds
- Inject summary into pool

---

## File Structure

```
src/
  exchange/
    __init__.py
    auditor.py      # Auditor LLM invocation
    messages.py     # Message type prefixes, formatting
hooks/
  auditor.py        # Auditor hook
  self_report.py    # Role injection hook
  restart_detector.py  # Restart signal detection
scripts/
  logos_chat.py     # Interactive mode (later)
```

---

## Suggested Order

1. **Task 1.1-1.3**: Auditor hook - get basic summarization working
2. **Task 2.1-2.2**: Self-reports - mutual legibility
3. **Task 3.1**: Manual observer response via inject
4. **Task 4.1-4.2**: Restart detection (basic)
5. **Task 5.1**: Cluster feed (optional enhancement)
6. **Task 3.2**: Interactive chat mode (when ready)

---

## Testing Strategy

Each phase testable independently:
- Phase 1: `logos config --set hooks='["auditor"]'` then `logos run 30`
- Phase 2: Add self_report hook, verify injections
- Phase 3: Manual inject after audit
- Phase 4: Inject "RESTART" messages, verify detection

No legacy code changes until protocol validated.
