# Vector DB & Attractor Analysis Plan

## Overview

Build instrumentation layer for analyzing memetic dynamics at scale. Enable systematic detection, comparison, and querying of attractor basins across experiments.

## Current State

**Existing capabilities (analyze.py):**
- Embedding generation (text-embedding-3-small via OpenRouter)
- Caching as .npz files per experiment
- Basic diversity metrics:
  - Within-round similarity (convergence indicator)
  - Distance from seed centroid (drift measure)
  - Output volume over time

**Limitations:**
- No attractor detection or clustering
- No cross-experiment comparison
- No semantic querying ("find messages about X")
- No efficient similarity search (won't scale)
- No temporal basin tracking

## Baseline Findings (Context)

All 4 models tested (Intellect-3, MIMO, Nemotron, Haiku) show:
1. **Output volume decay**: 4-8 msgs/round → 1-3 msgs/round by round 200
2. **Convergence to repetition**: Within-round similarity 0.4 → 0.6-1.0 (frequent identical messages)
3. **Semantic drift**: Distance from seed 0.43 → 0.55-0.67
4. **Attractor lock-in**: Pool becomes self-reinforcing homogeneous state

**Research questions:**
- What are the actual attractor messages? Meaningful or degenerate?
- When does convergence start? Phase transition?
- How do basins differ across models?
- What interventions could maintain diversity?

## Architecture

### Layer 1: Vector Storage (`src/analysis/vector_db.py`)

**VectorStore class** - Unified interface for similarity search

```python
class VectorStore:
    def __init__(self, index_type='faiss-flat'):
        """
        index_type: 'faiss-flat' (exact), 'faiss-ivf' (approximate, faster)
        """

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> list[str]:
        """
        Add vectors with metadata.

        metadata fields:
        - experiment: str
        - round: int
        - message: str
        - mind_id: int (optional)
        - timestamp: str (optional)

        Returns: list of vector IDs
        """

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[dict]:
        """
        Nearest neighbor search.

        Returns: [{id, distance, metadata}, ...]
        """

    def search_filtered(self, query_vector: np.ndarray,
                       filter_fn: Callable, top_k: int = 10) -> list[dict]:
        """
        Filtered similarity search.

        filter_fn: lambda metadata: bool
        Example: lambda m: m['experiment'] == 'baseline-haiku' and m['round'] > 100
        """

    def load_experiment(self, exp_dir: Path) -> int:
        """
        Load experiment embeddings into store.

        Reads: exp_dir/embeddings.npz
        Parses: exp_dir/logs/experiment.jsonl for metadata

        Returns: number of vectors added
        """

    def save(self, path: Path):
        """Save index to disk."""

    @classmethod
    def load(cls, path: Path) -> 'VectorStore':
        """Load index from disk."""
```

**Backend: FAISS**
- Handles 10M+ vectors on laptop
- Exact search (flat index) for initial development
- Can upgrade to approximate (IVF/HNSW) for scale
- Pure Python, no external services needed

### Layer 2: Attractor Detection (`src/analysis/attractors.py`)

**AttractorDetector class** - Identify and analyze basins

```python
class AttractorDetector:
    def __init__(self, vector_store: VectorStore):
        """Initialized with vector store containing experiment data."""

    def detect_attractors(self, experiment: str,
                         algorithm='hdbscan',
                         min_cluster_size=5) -> dict:
        """
        Cluster messages to identify attractors.

        Returns:
        {
            'clusters': [
                {
                    'id': int,
                    'size': int,
                    'centroid': np.ndarray,
                    'members': list[str],  # vector IDs
                    'rounds': list[int],   # when members appeared
                    'coherence': float     # avg intra-cluster similarity
                },
                ...
            ],
            'noise': list[str],  # unclustered vectors
            'algorithm': str,
            'params': dict
        }
        """

    def get_representative_messages(self, cluster_id: int,
                                   top_n: int = 10) -> list[dict]:
        """
        Extract most central messages from cluster.

        Returns messages closest to cluster centroid, with metadata.
        """

    def track_emergence(self, experiment: str,
                       window_size: int = 10) -> dict:
        """
        Track when attractors form over time.

        Returns time series of clustering state (entropy, cluster counts, etc.)
        """

    def compare_basins(self, exp1: str, exp2: str) -> dict:
        """
        Compare attractor basins between experiments.

        Returns:
        {
            'overlap': {
                'jaccard': float,  # cluster membership overlap
                'semantic': float  # centroid similarity
            },
            'unique_to_exp1': list[int],  # cluster IDs
            'unique_to_exp2': list[int],
            'shared': list[(int, int)],   # (exp1_cluster, exp2_cluster) pairs
            'basin_diagram': dict         # for visualization
        }
        """
```

**Clustering algorithm: HDBSCAN**
- Density-based (finds natural clusters)
- Auto-determines number of clusters
- Handles noise (doesn't force all points into clusters)
- Better than k-means for attractor detection

### Layer 3: Analysis Tools (`scripts/analyze.py` extensions)

**New analysis commands:**

```bash
# Detect attractors in single experiment
python scripts/analyze.py baseline-haiku --tool attractors

# Compare basins across models
python scripts/analyze.py baseline-haiku --tool compare-basins \
    --others baseline-mimo-flash,baseline-nemotron,baseline-intellect-3

# Query: find messages similar to a concept
python scripts/analyze.py baseline-haiku --tool query \
    --text "cooperation and transmission"

# Track convergence dynamics
python scripts/analyze.py baseline-haiku --tool convergence-timeline
```

**Output formats:**
- JSON (metrics, cluster assignments)
- YAML (representative messages)
- PNG (visualizations: basin diagrams, emergence timelines)
- CSV (time series data for external analysis)

## Implementation Phases

### Phase 1: Vector Store Foundation (MVP)
**Files:** `src/analysis/vector_db.py`

- VectorStore class with FAISS backend
- Load experiment embeddings from existing .npz files
- Basic search (exact nearest neighbor)
- Metadata filtering
- Save/load index

**Validation:**
- Load all 4 baseline experiments
- Query: "find messages about replication"
- Verify results make semantic sense

**Duration:** ~2 hours

### Phase 2: Attractor Detection
**Files:** `src/analysis/attractors.py`

- AttractorDetector class
- HDBSCAN clustering
- Representative message extraction
- Temporal tracking (when do basins form?)

**Validation:**
- Detect attractors in baseline-haiku
- Extract top-10 representative messages per cluster
- Verify clusters are semantically coherent

**Duration:** ~3 hours

### Phase 3: Cross-Experiment Comparison
**Extend:** `src/analysis/attractors.py`

- Basin comparison logic
- Overlap metrics (Jaccard, semantic similarity)
- Unique vs shared attractors
- Export comparison data

**Validation:**
- Compare baseline-haiku vs baseline-mimo-flash
- Identify model-specific attractors
- Quantify basin overlap

**Duration:** ~2 hours

### Phase 4: Analysis Commands & Outputs
**Extend:** `scripts/analyze.py`

- Register new tools: attractors, compare-basins, query, convergence-timeline
- Output formatting (JSON, YAML, visualizations)
- Documentation and examples

**Validation:**
- Run full analysis pipeline on baseline data
- Generate report comparing all 4 models
- Verify outputs are interpretable

**Duration:** ~2 hours

### Phase 5: Scale Testing & Optimization
**Refinement across codebase**

- Test with larger experiments (1000+ rounds, 10+ models)
- Benchmark search performance
- Optimize clustering parameters
- Add approximate search (IVF) if needed

**Duration:** ~2 hours

## Scale Considerations

**Current scale:** ~4000 messages total (4 experiments × 200 rounds × ~5 msgs/round)

**Future scale targets:**
- 100 experiments
- 1000 rounds each
- Variable message volumes
- **Est. 100k-500k messages**

**FAISS performance:**
- Flat index (exact): Handles 1M vectors easily on laptop
- IVF index (approximate): 10M+ vectors, <100ms queries
- Memory: ~4 bytes per dimension per vector (1536-dim × 100k = ~600MB)

**Storage strategy:**
- Embeddings: .npz per experiment (keep current caching)
- Centralized index: Build on-demand for cross-experiment analysis
- Option: Persist global index, incremental updates

## Dependencies

**New:**
- `faiss-cpu` (vector search)
- `hdbscan` (clustering)
- `umap-learn` (optional: dimensionality reduction for visualization)

**Already have:**
- numpy
- scikit-learn
- matplotlib

## Migration from Current System

**Backward compatible:**
- Keep existing embeddings tool (still useful standalone)
- Vector DB reads existing .npz files (no regeneration needed)
- Analysis tools are additive (novel-memes, embeddings still work)

**Transition:**
- Phase 1-2: Can use immediately with baseline data
- No changes needed to experiment runner
- Old analysis still works, new tools available

## Future Extensions

**Not in scope for MVP, but enabled by architecture:**

1. **Real-time attractor detection** - Monitor during experiment, trigger interventions
2. **Diversity injection** - Sample from under-represented clusters
3. **Multi-experiment pools** - Cross-pollinate between runs
4. **Attractor stability analysis** - Track basin persistence across runs
5. **Semantic search UI** - Web interface for querying message space
6. **Export to external tools** - Qdrant/Weaviate for distributed analysis

## Success Criteria

**Phase 1-2 (MVP):**
- Can detect attractors in baseline experiments
- Can extract top-N representative messages per attractor
- Attractors are semantically interpretable

**Phase 3-4 (Full):**
- Can compare basins across all 4 baseline models
- Can identify model-specific vs shared attractors
- Can query message space semantically
- Can track convergence dynamics over time

**Phase 5 (Production-ready):**
- Handles 100k+ messages efficiently (<1s queries)
- Clear documentation and examples
- Reproducible analysis pipeline

## Next Steps

1. ✅ Review this plan
2. ✅ Write to file
3. Implement Phase 1 (VectorStore + FAISS)
4. Test on baseline data
5. Iterate based on findings
