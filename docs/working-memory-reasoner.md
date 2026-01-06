# Working Memory Pool: Project Plan

*Reasoning as memetic ecology*

---

## Core Insight

Standard reasoning: sequential trace accumulation (chain-of-thought)
This project: **parallel thought ecology** with selection dynamics

The reasoning isn't in the generation - it's in the pool. Thoughts compete for attention. Persistence requires transmission. Forgetting is feature, not bug.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    REASONING LOOP                        │
│  while not terminated:                                   │
│    1. Sample K thoughts from pool                        │
│    2. Generate response (thinking + transmissions)       │
│    3. Add transmissions to pool                          │
│    4. Check termination conditions                       │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │   POOL    │  ← working memory
                    │  (FIFO)   │  ← tail M active
                    └───────────┘
                          │
                          ▼
                    ┌───────────┐
                    │  OUTPUT   │  ← final answer extraction
                    └───────────┘
```

---

## Phase 1: Minimal Viable Reasoner

**Goal:** Single-problem reasoning with pool-based working memory

### 1.1 Core Components

Reuse from Logosphere:
- `vector_db.py` - replaces Pool (Pool deleted as of Phase D)
- `mind.py` - minor modifications for reasoning context
- `logger.py` - unchanged

New:
- `reasoner.py` - reasoning loop orchestration
- `problem.py` - problem specification and answer extraction

### 1.2 Reasoner Loop

```python
class Reasoner:
    def __init__(self, pool: Pool, config: ReasonerConfig):
        self.pool = pool
        self.config = config
        self.iterations = 0
        
    def solve(self, problem: str, max_iterations: int = 50) -> str:
        """
        Run reasoning loop until termination.
        
        Returns extracted answer.
        """
        # Seed pool with problem
        self.pool.add_message(f"[PROBLEM] {problem}")
        
        while self.iterations < max_iterations:
            # Sample K thoughts
            thoughts = self.pool.sample(self.config.k_samples)
            
            # Invoke mind
            result = invoke_mind(
                system_prompt=REASONER_SYSTEM_PROMPT,
                messages=thoughts,
                token_limit=self.config.token_limit
            )
            
            # Add transmissions to pool
            for msg in result['transmitted']:
                self.pool.add_message(msg)
            
            # Check for answer
            answer = self.extract_answer(result['transmitted'])
            if answer:
                return answer
                
            self.iterations += 1
        
        # Max iterations - extract best answer from pool
        return self.extract_final_answer()
```

### 1.3 System Prompt (Draft)

```
You are reasoning through a problem. You receive thoughts from your working memory.

Read the thoughts. Think privately. Then transmit thoughts back to working memory.

Transmitted thoughts persist and may be sampled again. Thoughts not transmitted are forgotten.

When you have an answer, transmit: [ANSWER] your answer here

Format: Thoughts separated by --- on its own line.
```

### 1.4 Termination Conditions

- Explicit: `[ANSWER]` tag detected in transmission
- Implicit: Convergence detection (same answer transmitted N times)
- Fallback: Max iterations reached → extract from pool

### 1.5 Deliverables

- [ ] `reasoner.py` - core loop
- [ ] `problem.py` - problem loading, answer extraction
- [ ] `config_reasoner.py` - reasoner-specific config
- [ ] `test_reasoner.py` - basic functionality tests
- [ ] `run_reasoner.py` - entry point

---

## Phase 2: Emergent Markers

**Goal:** Observe and enable emergent prioritization

### 2.1 Hypothesis

Without explicit instruction, the system will evolve markers for:
- Importance: `[IMPORTANT]`, `[KEY]`, `!!!`
- Uncertainty: `[?]`, `[UNSURE]`, `[CHECK]`
- Progress: `[STEP 1]`, `[PARTIAL]`, `[BUILDING ON...]`
- Rejection: `[WRONG]`, `[DISCARD]`, `[SUPERSEDED]`

### 2.2 Observation Infrastructure

Track marker emergence across problems:

```python
def analyze_markers(pool_history: list[str]) -> dict:
    """
    Detect emergent patterns in pool content.
    
    Returns:
        - bracket_tags: [TAG] patterns and frequencies
        - repetition_patterns: phrases that get re-transmitted
        - compression_ratio: thought length over iterations
    """
```

### 2.3 Optional: Weighted Sampling

If markers emerge, test whether explicit weighting helps:

```python
def weighted_sample(pool: Pool, k: int, weights: dict) -> list[str]:
    """
    Sample with weights based on detected markers.
    
    E.g., [IMPORTANT] tagged messages get 2x weight.
    """
```

**Key principle:** Let markers emerge first, then optionally amplify. Don't design the vocabulary.

### 2.4 Deliverables

- [ ] `marker_analysis.py` - detect emergent patterns
- [ ] Logging infrastructure for marker tracking
- [ ] Optional weighted sampling (off by default)

---

## Phase 3: Benchmarking

**Goal:** Compare against standard chain-of-thought

### 3.1 Test Problems

Start with problems where parallel exploration helps:

1. **Multi-step math** - where dead ends are common
2. **Logic puzzles** - constraint satisfaction
3. **Planning problems** - where branching helps
4. **Ambiguous problems** - where multiple interpretations compete

### 3.2 Baselines

- Standard CoT (single sequential trace)
- Self-consistency (multiple CoT, vote on answer)
- Tree-of-thought (explicit branching)

### 3.3 Metrics

- Accuracy (correct answer rate)
- Efficiency (tokens to solution)
- Robustness (variance across runs)
- Emergence (marker diversity, compression)

### 3.4 Deliverables

- [ ] `benchmark.py` - run comparisons
- [ ] Problem sets (start small, curated)
- [ ] Results analysis and visualization

---

## Phase 4: Extensions (Deferred)

Ideas to explore after basic system works:

### 4.1 Adaptive Parameters

- K (samples) varies based on pool state
- M (active pool) grows/shrinks with problem complexity
- Token limit adapts to reasoning phase

### 4.2 Multi-Scale Pools

Nested pools for different timescales:
- Fast pool: immediate working memory (small M)
- Slow pool: persistent insights (large M, slower decay)
- Meta pool: strategies that work across problems

### 4.3 Collaborative Reasoning

Multiple "minds" reasoning in parallel:
- Shared pool (Logosphere-style)
- Different system prompts (critic, generator, checker)
- Emergent division of labor

### 4.4 Self-Modifying Prompts

The system transmits not just thoughts but prompt modifications:
- "When sampling, prefer thoughts tagged [VERIFIED]"
- Pool contains both content and meta-instructions

---

## Implementation Order

```
Week 1: Phase 1 (Minimal Viable Reasoner)
  Day 1-2: Core loop, reuse Logosphere components
  Day 3-4: System prompt iteration, termination logic
  Day 5: Testing on simple problems
  
Week 2: Phase 2 (Emergent Markers)
  Day 1-2: Observation infrastructure
  Day 3-4: Run experiments, analyze emergence
  Day 5: Document findings
  
Week 3: Phase 3 (Benchmarking)
  Day 1-2: Baseline implementations
  Day 3-4: Run comparisons
  Day 5: Analysis and writeup
```

---

## Key Design Decisions

### Pool vs. Context Window

The pool is NOT the context window. The context window contains:
- System prompt
- Sampled thoughts (K messages)
- Space for generation

The pool is external storage that persists across iterations. This is the key architectural difference from standard reasoning.

### FIFO vs. Other Decay

Start with FIFO (simplest). Could explore:
- LRU (least recently sampled drops)
- Importance-weighted decay
- No decay (pool grows indefinitely, random sample from all)

FIFO is the null hypothesis. Change only if data suggests otherwise.

### Single Mind vs. Ensemble

Start with single mind. The "parallel thoughts" are in the pool, not in multiple simultaneous generations. This is cheaper and cleaner.

Ensemble (multiple minds per iteration) is Phase 4 territory.

---

## Success Criteria

**Phase 1:** System runs, produces answers, logs are readable

**Phase 2:** Observable patterns emerge (markers, compression, something)

**Phase 3:** Competitive with or better than CoT on suitable problems

**Exciting:** Emergent phenomena we didn't design - the system develops its own cognitive vocabulary

---

## Connection to Logosphere

This is Logosphere turned inward. Same core insight:

> Memes don't exist in minds; they exist in transmission.

Becomes:

> Thoughts don't exist in reasoning steps; they exist in working memory transmission.

The pool is the pressure. Attention scarcity is the constraint. What cognitive structure arises?

---

## Open Questions

1. **Optimal K/M ratio** - How many thoughts to sample? How big should active memory be?

2. **Termination detection** - How does the system know it's done? Explicit markers? Convergence? Confidence?

3. **Problem seeding** - Just the problem statement? Or seed with relevant heuristics?

4. **Marker bootstrapping** - Do we hint at possible markers? Or pure emergence?

5. **Failure modes** - What happens when the pool fills with garbage? Self-recovery?

---

## Infrastructure Update (Jan 2026)

**RTD session completed Phases A-C.** The following infrastructure is now available:

### Available Components

| Component | File | What It Provides |
|-----------|------|------------------|
| **VectorDB** | `src/core/vector_db.py` | Replaces Pool. Same active pool semantics + embeddings + similarity search |
| **EmbeddingClient** | `src/core/embedding_client.py` | OpenRouter API wrapper, batch embedding, abort-on-failure |
| **AttractorDetector** | `src/analysis/attractors.py` | HDBSCAN clustering on active pool, finds semantic convergence |
| **Interventions** | `src/core/interventions.py` | Pluggable sampling hooks, registry pattern |
| **Logger extensions** | `src/core/logger.py` | `log_attractor_state`, `log_embedding_batch`, `log_error` |
| **Config extensions** | `src/config.py` | Embedding, attractor, intervention defaults |

### Adjustments to Approach

**Original plan:** Reuse Pool directly, minimal changes.

**Updated plan:** Use VectorDB instead. Key benefits:

1. **Convergence detection for free** - AttractorDetector can detect when thoughts converge to attractors. This is a natural termination condition:
   ```python
   # Instead of explicit [ANSWER] tag detection:
   attractor_state = detector.detect(iteration)
   if attractor_state['num_clusters'] == 1 and attractor_state['clusters'][0]['coherence'] > 0.8:
       # Thoughts have converged - extract answer from dominant cluster
   ```

2. **Semantic similarity for marker detection** - Embeddings let us find structurally similar thoughts without string matching:
   ```python
   # Find thoughts similar to "[IMPORTANT] ..."
   similar = vector_db.search_similar(marker_embedding, k=10)
   ```

3. **Weighted sampling via Intervention** - The hooks pattern is exactly what we need for Phase 2 (marker-weighted sampling):
   ```python
   class MarkerWeightedIntervention(Intervention):
       def on_sample(self, k, round_num):
           # Boost [IMPORTANT] tagged, reduce [DISCARD] tagged
           return self.vector_db.sample_weighted(k, weights)
   ```

4. **Unified persistence** - VectorDB save/load means reasoning sessions can be paused and resumed, analyzed post-hoc.

### Phase 1 Changes

**Before (original plan):**
- `reasoner.py` uses Pool
- Manual termination detection (string matching for [ANSWER])
- No embeddings

**After (updated - Pool deleted in Phase D):**
- `reasoner.py` uses VectorDB (Pool no longer exists)
- Termination via AttractorDetector (convergence = answer)
- Embeddings enabled (small overhead, big analysis value)
- Can reuse EmbeddingClient directly

**Phase 1.1 Deliverables:**
- [x] `src/reasoning/reasoner.py` - core loop using VectorDB
- [x] `scripts/run_reasoner.py` - entry point
- [ ] `problem.py` - problem loading from files (deferred - inline works)
- [ ] `test_reasoner.py` - formal tests (deferred)

### Implementation Status

**Phase 1.1 COMPLETE** (Jan 2026)

RTD session completed Phase D - Pool deleted, VectorDB is sole storage. This session validated the infrastructure by building a working reasoner on top.

Tested:
- `python scripts/run_reasoner.py "What is 23 + 47?"` → 70 (1 iteration)
- `python scripts/run_reasoner.py "A farmer has 17 chickens and 23 cows..."` → 126 legs (1 iteration)

Observation: Claude solves simple problems in 1 iteration (no pool dynamics visible). Need harder problems or constrained prompts to observe emergent markers and convergence.

---

## Next Action

**Phase 2: Observe emergence on harder problems**

1. Test on problems requiring exploration (logic puzzles, ambiguous problems)
2. Track marker emergence across sessions
3. Consider: constrained system prompt to force more deliberation?

Alternative: IDE integration for prompt decomposition (original use case).

---

*The reasoning is in the transmission.*