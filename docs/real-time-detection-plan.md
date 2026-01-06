# Real-Time Attractor Detection & Intervention Infrastructure

## Overview

Integrate real-time embedding generation, vector storage, and attractor detection into the experiment orchestrator to enable testing of dynamic intervention strategies (anti-convergence sampling, diversity injection, basin steering).

**Core architectural insight**: **VectorDB replaces Pool as the single source of truth** for all message data (text, embeddings, metadata). This eliminates redundancy between Pool/VectorStore/MessageStore and provides:
- Unified interface for sampling, search, clustering, and analysis
- Single data structure used by orchestrator, attractor detector, and interventions
- Perfect persistence: save VectorDB = complete experiment state
- Parallel analysis: load VectorDB while experiment runs
- Cluster only **active pool (M messages)** for constant O(M) performance

## Implementation Strategy: Hybrid Validation Approach

**Decision: All-in on unified VectorDB architecture with validation checkpoints**

Rather than building parallel systems (post-hoc VectorStore + runtime Pool), we're implementing the complete VectorDB replacement immediately. This reduces technical debt and establishes proper foundations while the project is still early.

**Risk mitigation through staged validation:**

1. **Build VectorDB with dual-mode support**
   - Runtime mode: `add(text, embedding, ...)` for live experiments
   - Post-hoc mode: `load_from_embeddings(npz_file, jsonl_logs)` for existing data

2. **Validate on baseline experiments BEFORE orchestrator integration**
   - Load existing .npz embeddings into VectorDB
   - Run clustering on 4 baseline experiments
   - Verify attractors are semantically coherent
   - Tune HDBSCAN parameters with real data

3. **Build supporting components in isolation**
   - EmbeddingClient (unit test with mocks)
   - AttractorDetector (validate on baseline VectorDBs)
   - Interventions (NoIntervention only for Phase 1)

4. **Integration testing before production**
   - Minimal 2-round experiment with embeddings enabled
   - Verify clean abort on API failure
   - Check attractor_detected events in logs

5. **Orchestrator refactoring last**
   - Replace Pool with validated VectorDB
   - All components proven to work independently

**Validation gates (must pass before proceeding):**
- ✅ VectorDB loads baseline embeddings correctly
- ✅ Clustering produces interpretable attractor messages
- ✅ Active pool tracking maintains correct FIFO window
- ✅ Search and sampling return expected results
- ✅ 2-round test experiment completes successfully

---

## Scope: Phase 1 - Detection Infrastructure Only

**What we're building:**
- ✅ **VectorDB** - Single source of truth replacing Pool/VectorStore/MessageStore
- ✅ Real-time embedding generation (batch at round boundary, abort on failure)
- ✅ Attractor detection (HDBSCAN clustering on active pool every round)
- ✅ Intervention hooks (infrastructure only - no concrete strategies yet)
- ✅ Unified storage and sampling interface

**What we're NOT building (deferred to Phase 2):**
- ❌ Concrete intervention strategies (e.g., anti-convergence sampling)
- ❌ Cluster-aware sampling (requires embeddings for entire active pool)
- ❌ Real-time visualization or monitoring dashboard

**Experimental boundary preserved:**
- Minds CANNOT see: Round numbers, timestamps, authorship, cluster labels
- Orchestrator CAN see: Embeddings, attractor basins, convergence metrics
- Interventions affect: Sampling strategy (what gets sampled), not Mind perception

## Architecture

### Data Flow Through Orchestrator Loop

```
Round Start
  ↓
For each Mind:
  1. Sample K messages from VectorDB (via intervention.on_sample() hook)
  2. Invoke Mind (LLM API)
  3. Parse output (thinking + transmitted)
  ↓
Batch embed all transmitted messages (1 API call per round)
  → Abort experiment on failure
  ↓
Add messages + embeddings → VectorDB with metadata
  ↓
Cluster ACTIVE POOL embeddings from VectorDB (not full pool!)
  → Update attractor state
  ↓
Notify intervention.on_round_end(attractor_state)
  ↓
Log round_end (reference vector IDs)
  ↓
Save VectorDB to disk (incremental or at experiment end)
```

### Key Design Decisions

1. **Single source of truth**: VectorDB replaces Pool, VectorStore, and MessageStore
2. **Embeddings**: Batched at round boundary (not per-message) for API efficiency
3. **Failure mode**: Abort experiment on embedding failure (clean failure > corrupted data)
4. **Clustering scope**: Active pool only (O(M) not O(N)) for constant performance
5. **Clustering frequency**: Every round (continuous detection)
6. **VectorDB persistence**: Save to disk incrementally or at end; enables parallel/post-hoc analysis
7. **Interventions**: Pluggable base class, Phase 1 implements `NoIntervention` only

## File Structure Changes

### New Files to Create

**Core infrastructure:**
- `src/core/vector_db.py` - **VectorDB class (replaces Pool, VectorStore, MessageStore)**
- `src/core/embedding_client.py` - OpenRouter API wrapper with abort-on-failure
- `src/core/interventions.py` - Base class + NoIntervention + factory

**Analysis infrastructure:**
- `src/analysis/attractors.py` - HDBSCAN clustering on active pool subset (queries VectorDB)

### Files to Modify

**orchestrator.py** (`src/core/orchestrator.py:1-98`)
- Replace `self.pool = Pool()` with `self.vector_db = VectorDB()`
- Replace all `pool.sample()` calls with `vector_db.sample_random()`
- Replace `pool.add_message()` with `vector_db.add()`
- Add intervention hooks around sampling (line 63)
- Add embedding batch generation (after line 90, before round_end)
- Add attractor detection call (cluster active pool)
- Add intervention notification hooks
- Modify logging to reference vector IDs instead of full text

**logger.py** (`src/core/logger.py:68-98, 100-113`)
- Modify `log_mind_invocation()` to accept vector IDs instead of full text
- Modify `log_round_end()` to accept vector IDs instead of pool_delta
- Add `log_attractor_state()` for clustering results
- Add `log_embedding_batch()` for performance tracking
- Add `log_error()` for abort scenarios

**pool.py** (`src/core/pool.py`)
- **DEPRECATED**: Pool class no longer used (replaced by VectorDB)
- Keep file for backward compatibility with old experiments
- New experiments use VectorDB instead

**config.py** (`src/config.py:27-32`)
- Add config defaults for embeddings, attractor detection, interventions
- Add per-experiment config loading (not just module-level constants)

## Implementation Details

### 1. VectorDB - Single Source of Truth

**Purpose**: Replaces Pool, VectorStore, and MessageStore with unified data structure

**Key methods:**
```python
class VectorDB:
    def __init__(self, active_pool_size: int):
        """Initialize with FAISS index + metadata tracking."""
        self.active_pool_size = active_pool_size  # M
        self.index = faiss.IndexFlatL2(1536)  # FAISS index
        self.metadata = []  # List of dicts per vector
        self.message_count = 0

    def add(self, text: str, embedding: np.ndarray,
            round_num: int, mind_id: int) -> int:
        """
        Add message with embedding. Returns vector ID.

        Replaces: pool.add_message() + vector_store.add() + message_store.add()
        """

    def sample_random(self, k: int, from_active_pool: bool = True) -> list[str]:
        """
        Sample k messages uniformly at random.

        Replaces: pool.sample()

        Returns: List of message texts
        """

    def sample_weighted(self, k: int, weights: dict[int, float]) -> list[str]:
        """
        Weighted sampling by vector ID (for interventions).

        Args:
            weights: {vector_id: weight} - boost/reduce sampling probability

        Returns: List of message texts
        """

    def get_active_pool_data(self) -> tuple[np.ndarray, list[dict]]:
        """
        Get active pool embeddings + metadata for clustering.

        Returns: (embeddings array [M×1536], metadata list)
        """

    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> list[dict]:
        """
        Similarity search.

        Returns: [{vector_id, distance, text, metadata}, ...]
        """

    def size(self) -> int:
        """Total messages in history."""
        return self.message_count

    def active_size(self) -> int:
        """Current active pool size."""
        return min(self.message_count, self.active_pool_size)

    def save(self, path: Path) -> None:
        """
        Save to disk.

        Writes:
        - path/index.faiss (FAISS index)
        - path/metadata.jsonl (one entry per vector)
        """

    @classmethod
    def load(cls, path: Path, active_pool_size: int) -> 'VectorDB':
        """Load from disk for post-hoc analysis."""
```

**Metadata schema:**
```python
{
    'vector_id': int,
    'round': int,
    'mind_id': int,
    'text': str,
    'timestamp': str,  # ISO8601
    'in_active_pool': bool  # Updated each round
}
```

**Storage format** (`vector_db/`):
```
vector_db/
├── index.faiss          # FAISS binary index
└── metadata.jsonl       # One line per vector
```

**Active pool tracking**: Last M messages (by vector ID) are marked `in_active_pool=True`

**Performance**:
- Add: <0.1ms per vector
- Sample random: O(M) to filter active pool, O(k) to sample
- Search: <1ms for k=10 on 1000+ vectors
- Save/load: ~100ms for 1000 vectors

### 2. EmbeddingClient - API Wrapper

**Purpose**: Generate embeddings with abort-on-failure semantics

**API details:**
- Model: `openai/text-embedding-3-small` (1536 dimensions)
- Batch: All transmitted messages in round (typically 1-5 messages)
- Latency: ~1s per API call
- Error handling: Raise `EmbeddingAPIError` → orchestrator aborts experiment

**Key methods:**
```python
class EmbeddingClient:
    def embed_batch(self, messages: list[str]) -> np.ndarray:
        """Generate embeddings for batch. Raises on failure."""
```

### 3. AttractorDetector - Cluster Active Pool

**Purpose**: HDBSCAN clustering on active pool (M messages), not full history

**Key method:**
```python
class AttractorDetector:
    def detect(self, round_num: int) -> dict:
        """
        Cluster active pool embeddings.

        Returns:
        {
            'detected': bool,
            'round': int,
            'clusters': [
                {
                    'id': int,
                    'size': int,
                    'centroid': np.ndarray,
                    'coherence': float,
                    'rounds': list[int]  # When members were created
                },
                ...
            ],
            'num_clusters': int,
            'noise_count': int
        }
        """
```

**Algorithm**: HDBSCAN
- Density-based (finds natural clusters)
- Auto-determines cluster count
- Handles noise (doesn't force all points into clusters)
- Parameters: `min_cluster_size=5`

**Performance** (active pool only):
- M=200: ~200ms per round (constant throughout experiment)
- M=500: ~500ms per round
- **Scales with M, not with total messages**

### 4. Intervention Infrastructure

**Purpose**: Pluggable hooks for sampling strategies (Phase 1: base class only)

**Base class:**
```python
class Intervention(ABC):
    def on_sample(self, vector_db: VectorDB, k: int, round_num: int) -> list[str]:
        """Override sampling strategy."""

    def on_round_start(self, round_num: int, attractor_state: dict) -> None:
        """React to attractor state before round."""

    def on_round_end(self, round_num: int, attractor_state: dict,
                      vector_ids: list[int]) -> None:
        """React after round ends."""
```

**Phase 1 implementation:**
```python
class NoIntervention(Intervention):
    """Default: standard uniform sampling from active pool."""
    def on_sample(self, vector_db: VectorDB, k: int, round_num: int) -> list[str]:
        return vector_db.sample_random(k, from_active_pool=True)
```

**Factory:**
```python
def create_intervention(config: dict, vector_db: VectorDB = None) -> Intervention:
    """Create intervention from config. Phase 1: always returns NoIntervention."""
```

## Orchestrator Integration Points

**Modified `run_round()` structure:**

```python
def run_round(self, round_num: int) -> int:
    round_vector_ids = []
    self.logger.log_round_start(
        round_num=round_num,
        pool_size=self.vector_db.size(),
        active_pool_size=self.vector_db.active_size()
    )

    # NEW: Pre-round intervention hook
    if self.intervention:
        self.intervention.on_round_start(round_num, self.attractor_detector.get_state())

    for mind_id in range(self.n_minds):
        # MODIFIED: Sampling via intervention from VectorDB
        if self.intervention:
            sampled = self.intervention.on_sample(self.vector_db, self.k_samples, round_num)
        else:
            sampled = self.vector_db.sample_random(self.k_samples, from_active_pool=True)

        result = invoke_mind(
            system_prompt=self.system_prompt,
            messages=sampled,
            token_limit=self.token_limit
        )

        self.total_tokens += result['tokens_used']

        # Collect transmitted messages for batch embedding
        for msg in result['transmitted']:
            round_vector_ids.append((msg, mind_id))

        # Log thinking separately (not transmitted, not embedded)
        thinking_text = result['thinking']
        self.logger.log_thinking(round_num, mind_id, thinking_text)

    # NEW: Batch embed all transmitted messages from this round
    if round_vector_ids and self.embedding_client.enabled:
        message_texts = [msg for msg, _ in round_vector_ids]
        try:
            embeddings = self.embedding_client.embed_batch(message_texts)

            # Add to VectorDB (replaces pool.add + vector_store.add)
            vector_ids = []
            for (msg, mind_id), embedding in zip(round_vector_ids, embeddings):
                vid = self.vector_db.add(
                    text=msg,
                    embedding=embedding,
                    round_num=round_num,
                    mind_id=mind_id
                )
                vector_ids.append(vid)

            # Log mind invocations with vector IDs
            self.logger.log_mind_invocation(
                round_num=round_num,
                vector_ids=vector_ids,
                tokens_used=result['tokens_used']
            )
        except Exception as e:
            self.logger.log_error(f"Embedding failed: {e}", round_num)
            raise ExperimentAbortError(f"Round {round_num}: {e}") from e

    # NEW: Detect attractors (cluster active pool only!)
    if self.attractor_detector.enabled:
        attractor_state = self.attractor_detector.detect(round_num)
        self.logger.log_attractor_state(round_num, attractor_state)

    # NEW: Post-round intervention hook
    if self.intervention:
        self.intervention.on_round_end(round_num, attractor_state, vector_ids)

    # Log round end
    self.logger.log_round_end(
        round_num=round_num,
        messages_added=len(vector_ids),
        vector_ids=vector_ids
    )

    # Save VectorDB incrementally (optional, could do at end only)
    if round_num % 10 == 0:  # Save every 10 rounds
        self.vector_db.save(self.output_dir / "vector_db")

    return len(vector_ids)
```

**Key integration points:**
- **Line 1-7**: Replace `pool.size()` calls with `vector_db.size()`
- **Line 14-17**: Replace `pool.sample()` with `vector_db.sample_random()`
- **Line 23-25**: Collect messages for batch embedding (no immediate add)
- **Line 31-48**: Batch embed, then add to VectorDB with embeddings
- **Line 51-52**: Detect attractors using VectorDB active pool
- **Line 55-56**: Intervention sees vector IDs, not message hashes
- **Line 59-63**: Log vector IDs instead of pool delta
- **Line 65-67**: Save VectorDB incrementally

## Configuration Schema

**Extend config.json:**
```json
{
  "parameters": {
    "N_MINDS": 1,
    "K_SAMPLES": 10,
    "M_ACTIVE_POOL": 200,
    "MAX_ROUNDS": 200,
    "TOKEN_LIMIT": 8000
  },
  "api": {
    "MODEL": "anthropic/claude-sonnet-4.5",
    "API_BASE_URL": "https://openrouter.ai/api/v1"
  },
  "system_prompt": "...",

  "embeddings": {
    "enabled": true,
    "model": "openai/text-embedding-3-small",
    "fail_mode": "abort"
  },

  "attractor_detection": {
    "enabled": true,
    "algorithm": "hdbscan",
    "min_cluster_size": 5,
    "cluster_active_pool_only": true
  },

  "interventions": {
    "enabled": false,
    "strategy": "none"
  }
}
```

**Defaults in config.py:**
```python
DEFAULT_EMBEDDING_CONFIG = {
    'enabled': False,  # Backward compatible: opt-in
    'model': 'openai/text-embedding-3-small',
    'fail_mode': 'abort'
}

DEFAULT_ATTRACTOR_CONFIG = {
    'enabled': False,
    'algorithm': 'hdbscan',
    'min_cluster_size': 5,
    'cluster_active_pool_only': True
}

DEFAULT_INTERVENTION_CONFIG = {
    'enabled': False,
    'strategy': 'none'
}
```

## Error Handling

**Embedding API failure:**
```python
# In orchestrator
except Exception as e:
    self.logger.log_error(f"Embedding failed: {e}", round_num)
    print(f"\n❌ EXPERIMENT ABORTED: Embedding API failure in round {round_num}")
    print(f"   Completed rounds: {round_num - 1}")
    print(f"   Logs: {self.logger.log_file}")
    raise ExperimentAbortError(...) from e
```
- **Result**: Clean failure, partial results (rounds 1 to N-1) fully logged and usable

**Clustering failure:**
```python
# In AttractorDetector.detect()
except Exception as e:
    logger.warning(f"Clustering failed: {e}")
    return {'detected': False, 'error': str(e), 'clusters': []}
```
- **Result**: Warning logged, experiment continues without attractor detection

**Intervention failure:**
```python
# In orchestrator
except Exception as e:
    logger.warning(f"Intervention failed, falling back to uniform sampling: {e}")
    sampled_messages = self.pool.sample(self.k_samples)
```
- **Result**: Revert to baseline sampling, experiment continues

## Performance Analysis

**Per-round overhead (Phase 1):**

| Component | Latency | Notes |
|-----------|---------|-------|
| Mind invocation | 3-6s | Unchanged (LLM API) |
| Embedding API | ~1s | 1 batch call per round (~5 msgs) |
| VectorStore updates | <10ms | Add embeddings, mark active pool |
| HDBSCAN (active pool) | ~200ms | **Constant for M=200, regardless of experiment length** |
| Intervention hooks | <10ms | NoIntervention in Phase 1 |
| Logging (hashes) | <10ms | Faster than before (no full text) |
| **Total** | **4.2-7.2s** | **~1.2s overhead over baseline** |

**Key insight**: Clustering active pool only (O(M)) provides constant performance:
- Round 10: 200ms clustering
- Round 100: 200ms clustering
- Round 1000: 200ms clustering
- **No scaling issues!**

## Logging Changes

**New file structure:**
```
experiments/<name>/
├── config.json
├── init.md
├── logs/
│   └── experiment.jsonl     # Event stream (references vector IDs)
└── vector_db/
    ├── index.faiss          # FAISS index (embeddings)
    └── metadata.jsonl       # Message text + metadata per vector
```

**Modified events:**
- `round_start`: Pool size from `vector_db.size()`
- `mind_invocation`: References vector IDs, not full text
- `round_end`: List of vector IDs added this round
- **Message content lives in** `vector_db/metadata.jsonl`, not experiment.jsonl

**New events:**
- `thinking`: Thinking content per Mind (not transmitted, logged separately)
- `attractor_detected`: Cluster summary (count, sizes, coherence)
- `embedding_batch`: Performance metrics (latency)
- `error`: Abort scenarios

**Post-hoc analysis**: Load `VectorDB.load(exp_dir / "vector_db")` to access all messages + embeddings

## Dependencies

**New Python packages:**
```bash
pip install hdbscan  # Only new dep needed
```

**Already have (via `uv sync --extra analysis`):**
- numpy, scikit-learn, matplotlib

**Scaling note:** VectorDB uses sklearn's NearestNeighbors (brute-force cosine) which is
sufficient for M=200 active pool (~10ms search). If scaling to 10k+ vectors, swap to
faiss-cpu or hnswlib - the VectorDB interface is designed for drop-in replacement.

## Testing Strategy

**Unit tests:**
- MessageStore: Hash stability, deduplication
- EmbeddingClient: Mock API, abort-on-failure
- VectorStore: Active pool tracking, FAISS operations
- AttractorDetector: Synthetic data clustering
- Interventions: Hook execution

**Integration test:**
- Minimal 2-round experiment with embeddings enabled
- Verify file structure (experiment.jsonl, messages.jsonl)
- Check attractor_detected events

**Validation:**
- Run baseline experiment with real-time infrastructure
- Compare attractor clusters to post-hoc analysis (should match for active pool)

## Implementation Order (Hybrid Validation)

**Phase A: Core Components + Baseline Validation**
1. ✅ **VectorDB** (`src/core/vector_db.py`)
   - sklearn-based (NearestNeighbors), swappable to FAISS if needed
   - Dual-mode support (runtime + post-hoc via `load_from_legacy()`)
   - Active pool tracking, random/weighted sampling, similarity search

2. ✅ **Validate on Baseline Data**
   - Loaded all 4 baseline experiments from .npz + JSONL
   - Verified active pool tracking, sampling, search, save/load roundtrip

3. ✅ **AttractorDetector** (`src/analysis/attractors.py`)
   - HDBSCAN clustering on active pool embeddings
   - Representative message extraction, coherence scoring
   - All 4 baselines → 2 clusters each, 0.74-0.78 coherence

4. ✅ **Baseline Attractor Analysis**
   - Validated on all 4 baseline experiments
   - Confirmed semantic coherence (AI-assistant acknowledgment patterns)

**Phase B: Runtime Components**
5. ✅ **EmbeddingClient** (`src/core/embedding_client.py`)
   - OpenRouter API wrapper (text-embedding-3-small)
   - Batch embedding with abort-on-failure (EmbeddingAPIError)
   - Validated with real API calls

6. ✅ **Interventions Base** (`src/core/interventions.py`)
   - Abstract base class with on_sample/on_round_start/on_round_end hooks
   - NoIntervention implementation (baseline sampling)
   - Registry + factory for future intervention strategies

**Phase C: Integration** ✅
7. ✅ **Modify Logger** (`src/core/logger.py`)
   - New: log_attractor_state, log_embedding_batch, log_error
   - Added optional vector_ids to log_round_end

8. ✅ **Extend Config** (`src/config.py`)
   - Added DEFAULT_EMBEDDING/ATTRACTOR/INTERVENTION_CONFIG
   - Added load_experiment_config() for merging defaults

9. ✅ **Integration Test** (`scripts/test_realtime.py`)
   - 2-round test with real API embedding calls
   - Verified: VectorDB, EmbeddingClient, AttractorDetector, Logger
   - All events logged correctly, save/load roundtrip works

**Phase D: Orchestrator Refactoring** ✅
10. ✅ **Modify Orchestrator** (`src/core/orchestrator.py`)
    - Replaced Pool with VectorDB
    - Integrated EmbeddingClient, AttractorDetector, Interventions
    - Added intervention hooks (on_sample, on_round_start, on_round_end)
    - ExperimentAbortError on embedding failure

11. ✅ **Delete Pool** (`src/core/pool.py`)
    - DELETED entirely (no backward compat, per design principle)
    - VectorDB is the single source of truth
    - Old code importing Pool gets ImportError (signal to update)

12. ✅ **Full Validation** (`scripts/run_validation_experiment.py`)
    - 10-round experiment with real LLM + embeddings + attractor detection
    - All logging events present (embedding_batch, attractor_state, etc.)
    - VectorDB save/load roundtrip verified
    - Validation passed 2026-01-06

## Success Criteria

**Phase 1 complete when:**
- ✅ VectorDB replaces Pool as single source of truth
- ✅ Experiments can run with real-time embeddings enabled
- ✅ Attractor detection runs every round on active pool (constant ~200ms)
- ✅ Logs show attractor_state events with cluster summaries
- ✅ VectorDB persists to disk (incremental or at end)
- ✅ Post-hoc analysis can load VectorDB and query messages
- ✅ Experiment aborts cleanly on embedding failure (ExperimentAbortError)
- ✅ Intervention hooks exist (even if NoIntervention is used)
- ~~Backward compatible: old experiments can still use Pool~~ → Pool deleted, no backward compat

**NOT required for Phase 1:**
- ❌ Concrete intervention strategies (Phase 2)
- ❌ Cluster-aware sampling (Phase 2)
- ❌ Real-time visualization (Phase 3)

## Estimated Effort

- **Phase 1**: 6-8 hours (1-2 days focused work)
- **Phase 2** (interventions): 4-6 hours
- **Phase 3** (monitoring/tuning): 2-4 hours

## Next Steps After Phase 1

**Phase 2: Anti-Convergence Sampling**
- Embed entire active pool (not just new messages) for cluster membership
- Implement weighted sampling (boost under-represented clusters)
- Validate diversity maintenance vs baseline

**Phase 3: Monitoring & Adaptive Tuning**
- Performance monitoring (clustering latency tracking)
- Adaptive update frequency (switch to every-N if needed)
- Optional visualization dashboard

## Critical Files

**Primary modifications:**
- `/home/daniel/prj/logosphere/src/core/orchestrator.py:1-98` - Replace Pool with VectorDB
- `/home/daniel/prj/logosphere/src/core/logger.py:68-113` - Log vector IDs instead of text
- `/home/daniel/prj/logosphere/src/config.py:27-50` - Add embedding/attractor/intervention config

**New files:**
- `/home/daniel/prj/logosphere/src/core/vector_db.py` - **VectorDB class (replaces Pool)**
- `/home/daniel/prj/logosphere/src/core/embedding_client.py` - Embedding API wrapper
- `/home/daniel/prj/logosphere/src/core/interventions.py` - Intervention base class
- `/home/daniel/prj/logosphere/src/analysis/attractors.py` - HDBSCAN clustering

**Deprecated:**
- `/home/daniel/prj/logosphere/src/core/pool.py` - Keep for backward compat, mark deprecated

---

**This plan enables the core research agenda: testing dynamic interventions to steer pool dynamics based on real-time attractor detection, while preserving experimental purity (Minds remain blind to all metadata/clustering).**
