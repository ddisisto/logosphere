# CLAUDE.md - Architecture & Principles

## Experimental Boundary

**INSIDE THE EXPERIMENT:**
- Messages in the pool
- Mind inputs (sampled messages + system prompt)
- Mind outputs (parsed messages)

**OUTSIDE THE EXPERIMENT:**
- Orchestrator (scheduler, sampler, parser)
- API calls to LLM
- Logging system
- File I/O
- Analysis tools

**Critical:** Nothing inside the experiment can see round numbers, timestamps, authorship, or any metadata. Messages are anonymous and timeless from the perspective of Minds.

This boundary is what makes memetic dynamics observable - the system evolves based purely on message content, not external coordination.

---

## Current Structure

```
logosphere/
├── src/
│   ├── core/              # Experiment engine
│   │   ├── vector_db.py   # Message storage with embeddings
│   │   ├── session.py     # Branch management, visibility, interventions
│   │   ├── intervention_log.py # Append-only JSONL audit trail
│   │   ├── embedding_client.py # OpenRouter embedding API
│   │   ├── interventions.py # Sampling strategy hooks
│   │   ├── mind.py        # API invocation and parsing
│   │   ├── orchestrator.py # Main loop, coordination
│   │   ├── logger.py      # Structured logging
│   │   └── init_parser.py # Seed message parsing
│   ├── logos/             # Logos reasoning CLI
│   │   ├── config.py      # Configuration + defaults
│   │   └── runner.py      # Core loop: sample → mind → embed → add
│   ├── analysis/          # Analysis tools
│   │   └── attractors.py  # HDBSCAN clustering for attractor detection
│   └── config.py          # Parameters and system prompt
├── scripts/
│   ├── run.py             # Experiment entry point
│   ├── logos.py           # Logos reasoning CLI
│   └── analyze.py         # Post-hoc analysis
├── tests/                 # Validation tests
└── experiments/           # Self-contained experiment runs
    └── <name>/
        ├── init.md        # Seed messages
        ├── config.json    # Parameter snapshot
        ├── logs/          # JSONL event stream
        └── vector_db/     # Embeddings + metadata
```

---

## Development Setup

```bash
uv sync                      # Install core dependencies
uv sync --extra analysis     # Add analysis tools (numpy, sklearn, matplotlib)
uv sync --extra dev          # Add dev tools (pytest)
```

Run experiments: `python scripts/run.py <experiment_name>`
Run logos: `python scripts/logos.py --help`
Run analysis: `python scripts/analyze.py <experiment_name> --tool <tool>`
Run tests: `pytest`

---

## Core Mechanics

### VectorDB (src/core/vector_db.py)

Message storage with embeddings and FIFO active window:

- **History**: All messages ever added (append-only) with embeddings
- **Active Pool**: Tail M messages (sampling window)
- **Sampling**: Uniform random from active pool (or weighted via interventions)
- **Search**: Similarity search via embeddings

```python
vector_db = VectorDB(active_pool_size=200)
vid = vector_db.add(text=msg, embedding=emb, round_num=1, mind_id=0)
sample = vector_db.sample_random(k=10)      # Sample from tail M
similar = vector_db.search_similar(query_emb, k=5)  # Semantic search
```

**Why tail sampling?** Creates recency bias without explicit timestamps. Recent messages more likely to be sampled, but old messages eventually fall out of active window. This is how "forgetting" happens.

**Why embeddings?** Enables real-time attractor detection (clustering), semantic search, and future intervention strategies (e.g., diversity-weighted sampling).

### Mind (src/core/mind.py)

Stateless LLM invocation:

- **Input**: System prompt + K sampled messages
- **Output**: Thinking (private) + transmitted messages (public)
- **Parsing**: Everything before first `---` is thinking, blocks between `---` are messages

```python
result = invoke_mind(
    system_prompt="You receive messages...",
    messages=sample,
    token_limit=8000
)
# result['thinking'] - not transmitted
# result['transmitted'] - added to pool
```

**Key:** Each Mind invocation is independent. No conversation history, no memory. Only the pool persists.

### Orchestrator (src/core/orchestrator.py)

Coordinates rounds with real-time detection:

1. Sample K messages from VectorDB (via intervention hook)
2. Invoke Mind with sample
3. Parse output (thinking vs transmitted)
4. Batch embed all transmitted messages
5. Add to VectorDB with embeddings
6. Detect attractors (cluster active pool)
7. Notify intervention of attractor state
8. Log everything
9. Repeat for N minds per round

**Sequential invocation:** Minds within a round see VectorDB state as it evolves. Mind 0 adds messages that Mind 1 might sample.

**Abort on failure:** Embedding API errors trigger ExperimentAbortError for clean failure (partial results preserved).

### Logos CLI (scripts/logos.py)

Pool-based reasoning with branch-based history management:

- **Branch-based sessions** - create branches from any state, switch between them
- **Minimal protocols** - external prompts prefixed with `>>> `
- **Pool state = output** - answer emerges from dominant cluster
- **Per-iteration metrics** - diversity, cluster count, coherence tracked
- **Intervention logging** - all actions recorded in append-only audit trail

```bash
python scripts/logos.py init ./session "initial prompt"   # Create session
python scripts/logos.py open ./session                    # Open existing
python scripts/logos.py run 10                            # Batch iterations
python scripts/logos.py step                              # Single iteration
python scripts/logos.py inject "thought text"             # Add external message
python scripts/logos.py branch experiment                 # Create branch
python scripts/logos.py switch main                       # Switch branch
python scripts/logos.py status                            # Current state
python scripts/logos.py list                              # Show all branches
python scripts/logos.py config                            # Show/set branch config
python scripts/logos.py log                               # Intervention history
```

**Session model:**
- Append-only VectorDB with branch field per message
- Visibility computed by filtering (no data duplication)
- `branches.json` stores `{name, parent, parent_iteration}` per branch
- Deep session copy via filesystem (`cp -rp`) for parallel experiments

**Philosophy:** Memes win by replication, not by declaring themselves important. The `>>> ` prefix is minimal metadata to distinguish external input from internal generation.

### Logging (src/core/logger.py)

JSONL event stream:

- `experiment_start` - config, seed count
- `round_start` - VectorDB state (total size, active pool size)
- `mind_invocation` - full I/O (input sample, thinking, transmitted, tokens)
- `embedding_batch` - embedding latency, model
- `attractor_state` - cluster count, sizes, coherence, representatives
- `round_end` - pool delta
- `experiment_end` - summary stats

**Why JSONL?** Streaming, line-at-a-time, easy to parse, survives interruptions.

---

## Experiment Configuration

Each experiment is self-contained in `experiments/<name>/`:

### config.json

```json
{
  "N_MINDS": 1,
  "K_SAMPLES": 10,
  "M_ACTIVE_POOL": 200,
  "MAX_ROUNDS": 200,
  "TOKEN_LIMIT": 8000,
  "MODEL": "anthropic/claude-sonnet-4.5",
  "SYSTEM_PROMPT": "You receive messages from others..."
}
```

**Key parameters:**
- `N_MINDS`: Minds per round (1 = single model, >1 = population)
- `K_SAMPLES`: Messages sampled per Mind (attention budget)
- `M_ACTIVE_POOL`: Active pool size (recency window)
- `MAX_ROUNDS`: Stopping condition
- `MODEL`: OpenRouter model ID

### init.md

Seed messages in Mind output format:

```
Optional notes here (not transmitted to pool)

---
First seed message
---
Second seed message
---
```

**Parsing:** Reuses Mind parsing logic. Everything before first `---` is ignored. Blocks between `---` are messages.

**Workflow:**
1. Copy from template experiment (e.g., `experiments/_baseline/init.md`)
2. Customize seed messages
3. Run experiment

---

## Analysis Pipeline

### Built-in Tools (scripts/analyze.py)

**novel-memes** - Extract all transmitted messages to YAML
```bash
python scripts/analyze.py <experiment> --tool novel-memes
```

**embeddings** - Generate embeddings and diversity metrics
```bash
python scripts/analyze.py <experiment> --tool embeddings
```

Produces:
- `embeddings.npz` - Cached vectors (text-embedding-3-small)
- `diversity_metrics.json` - Time series (similarity, drift, volume)
- `diversity_plots.png` - Visualizations

### Extensible Framework

Add new tools by:
1. Define `analyze_<toolname>(exp_dir: Path)` in `analyze.py`
2. Register in `TOOLS` dict
3. Done

See `docs/vector-db-plan.md` for upcoming attractor analysis tools.

---

## Observed Dynamics (Baseline Experiments)

**Setup:** 4 models, 200 rounds each, single Mind per round, dense philosophical seed content

**Universal patterns across all models:**
1. **Output volume decay**: 4-8 msgs/round → 1-3 msgs/round
2. **Convergence to repetition**: Within-round similarity 0.4 → 0.6-1.0 (frequent identical messages)
3. **Semantic drift**: Distance from seed 0.43 → 0.65+
4. **Attractor lock-in**: Pool becomes self-reinforcing homogeneous state

**Mechanism:** Sample similar messages → produce similar outputs → pool becomes more homogeneous → sampling becomes even less diverse → feedback loop to convergence.

**Open questions:**
- What are the actual attractor messages? (Need clustering to identify)
- When does convergence start? (Phase transition detection)
- How do basins differ across models? (Cross-experiment comparison)
- Can we intervene to maintain diversity? (Sample from under-represented clusters)

---

## Design Principles

### 1. Experimental Purity
Messages contain only content, no metadata. Minds cannot see:
- Round numbers
- Timestamps
- Authorship (who wrote what)
- How many times a message was sampled
- Anything outside message text

This forces pure memetic selection - messages persist/spread based on content alone.

### 2. Statefulness in Pool Only
- Minds are stateless (no memory between invocations)
- Pool is the only persistent state
- All information transmission happens through pool

This makes the pool the "culture" - the collective memory that evolves.

### 3. Observable at Every Level
- Full I/O logging (inputs, thinking, outputs)
- Streaming JSONL (survives crashes, real-time monitoring)
- Pool deltas not snapshots (see what changed each round)
- Embeddings cacheable (reuse across analyses)

### 4. Composable Analysis
- Experiments don't know about analysis
- Analysis tools are post-hoc, optional, extensible
- Same experiment can be analyzed multiple ways
- New tools don't require code changes to core engine

---

## Parameter Tuning Guidelines

**N_MINDS:**
- 1 = single model iteration (baseline)
- 2-5 = small population dynamics
- 10+ = ecosystem (expensive)

**K_SAMPLES:**
- Small (3-5) = narrow attention, strong selection pressure
- Medium (10-20) = balanced awareness
- Large (50+) = broad context, weaker selection

**M_ACTIVE_POOL:**
- Small (15-50) = rapid turnover, short-term memory
- Medium (100-200) = balanced recency bias
- Large (500+) = long memory, slower evolution

**Interactions:**
- K should be < M (sample from larger pool)
- M/K ratio controls selection pressure (higher = more diversity in active pool)
- N × msgs_per_mind controls pool growth rate

---

## Future Directions

**Recently implemented:**
- VectorDB for similarity search (replaces Pool)
- Attractor detection via HDBSCAN clustering
- Logos CLI with branch-based session management
- Intervention logging (append-only audit trail)
- Per-branch config tracking
- Real-time dynamics tracking (diversity, coherence, clusters)

**Potential experiments:**
- Heterogeneous populations (multiple models in same pool)
- Diversity interventions (anti-convergence sampling strategies)
- Inject pool state awareness as thoughts (not system prompt)
- Cross-pool pollination (message exchange between experiments)
- Small model dynamics comparison (haiku vs sonnet)

**Research questions:**
- Can we measure "meme fitness" independent of model?
- Do certain message structures replicate better?
- What determines basin depth (easy vs hard to escape)?
- Can we predict convergence from early dynamics?

---

## Success Criteria

**Working correctly:**
- Experimental boundary maintained (no metadata leakage)
- Pool mechanics correct (FIFO tail, uniform sampling)
- Logs complete and parseable
- Analysis tools produce interpretable results

**Scientifically interesting:**
- Pool dynamics are non-trivial (not random, not deterministic)
- Emergence is observable (patterns not in seed content)
- Cross-model comparison reveals differences
- Interventions produce measurable effects

**The goal:** Build a substrate where memetic dynamics can be observed, measured, and manipulated. If we can see ideas compete, cooperate, and evolve in the pool, we have the foundation for systematic study of cultural evolution.
