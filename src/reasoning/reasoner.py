"""
Working Memory Reasoner - Pool-based thought ecology.

Core philosophy: No explicit protocols. The pool state IS the output.

- No [ANSWER] tags - memes win by replication, not declaration
- Termination is dynamics-based (convergence, stability, timeout)
- We measure and steer the pool, not the answers
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json

import numpy as np

from src.core.vector_db import VectorDB
from src.core.embedding_client import EmbeddingClient
from src.core.mind import invoke_mind
from src.analysis.attractors import AttractorDetector


# Minimal system prompt - minimal protocols, just framing
REASONER_SYSTEM_PROMPT = """You receive thoughts from a shared pool.

Read them. Think privately. Transmit thoughts worth keeping.

Transmitted thoughts persist and compete for attention.
Thoughts not transmitted are forgotten.
Thoughts should be largely self-contained, but may reference or include others.
Thoughts may be retransmitted verbatim, and/or compressed, extended, decomposed, recomposed, etc.

Incoming prompts from external sources are prefixed with '>>> '.

Format: Thoughts (multi-line, unconstrained) separated by --- on its own line. e.g.

```
This is space for private thought and reasoning
---
This is the first transmitted thought
---
This is another
---
```
"""

# Prefix for externally-sourced prompts
EXTERNAL_PROMPT_PREFIX = ">>> "


@dataclass
class ReasonerConfig:
    """Configuration for reasoning session."""

    # Pool parameters
    k_samples: int = 5  # Thoughts sampled per iteration
    active_pool_size: int = 50  # Working memory capacity

    # Termination (dynamics-based)
    max_iterations: int = 30
    convergence_threshold: float = 0.75  # Coherence for convergence
    convergence_coverage: float = 0.50  # Cluster must cover this % of pool
    stability_window: int = 3  # Iterations of stable clusters to terminate
    min_cluster_size: int = 3  # For HDBSCAN
    no_early_termination: bool = False  # If True, run all iterations

    # LLM
    model: str = "anthropic/claude-haiku-4-5-20241022"
    token_limit: int = 4000
    system_prompt: str = REASONER_SYSTEM_PROMPT

    # Embeddings
    embedding_model: str = "openai/text-embedding-3-small"

    # Output
    output_dir: Optional[Path] = None
    verbose: bool = True


@dataclass
class IterationMetrics:
    """Metrics for a single iteration."""
    iteration: int
    pool_size: int
    active_size: int
    num_clusters: int
    dominant_cluster_size: int
    dominant_cluster_share: float
    coherence: float
    diversity: float  # Mean pairwise distance in active pool
    thoughts_added: int


@dataclass
class ReasoningResult:
    """Result of a reasoning session."""
    termination_reason: str  # 'converged', 'stable', 'timeout'
    iterations: int
    total_thoughts: int
    final_pool_size: int

    # Pool state output
    dominant_cluster_texts: list[str]  # Representative texts from dominant cluster
    cluster_count: int
    final_coherence: float

    # Trajectory data
    metrics_history: list[IterationMetrics] = field(default_factory=list)
    thinking_trace: list[str] = field(default_factory=list)


class Reasoner:
    """
    Pool-based reasoner using VectorDB for working memory.

    The reasoning loop:
    1. Sample K thoughts from working memory (VectorDB)
    2. Model thinks, transmits (no protocol)
    3. Embed new thoughts, add to VectorDB
    4. Measure dynamics (diversity, clusters, stability)
    5. Check termination (dynamics only - convergence, stability, timeout)
    6. Output = pool state (dominant cluster or trajectory)
    """

    def __init__(self, config: Optional[ReasonerConfig] = None):
        """Initialize reasoner with config."""
        self.config = config or ReasonerConfig()

        # Initialize components
        self.vector_db = VectorDB(
            active_pool_size=self.config.active_pool_size
        )
        self.embedding_client = EmbeddingClient(
            model=self.config.embedding_model,
            enabled=True
        )
        self.attractor_detector = AttractorDetector(
            vector_db=self.vector_db,
            min_cluster_size=self.config.min_cluster_size
        )

        # State
        self.iteration = 0
        self.thinking_trace: list[str] = []
        self.metrics_history: list[IterationMetrics] = []
        self.cluster_history: list[int] = []  # Track cluster count for stability
        self._prompts: list[str] = []  # External prompts, stored for resume

    @classmethod
    def from_checkpoint(
        cls,
        output_dir: Path,
        config_overrides: Optional[dict] = None
    ) -> "Reasoner":
        """
        Load a Reasoner from a saved checkpoint.

        Args:
            output_dir: Directory containing state.json, metrics.jsonl, vector_db/
            config_overrides: Optional dict to override loaded config values
                             (e.g., {"max_iterations": 100} to extend)

        Returns:
            Reasoner instance with restored state
        """
        output_dir = Path(output_dir)
        state_path = output_dir / "state.json"

        if not state_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: no state.json found in {output_dir}. "
                "This run predates resume support or is incomplete."
            )

        state = json.loads(state_path.read_text())

        # Restore config with overrides
        saved_config = state["config"]
        if config_overrides:
            saved_config.update(config_overrides)

        # Convert output_dir back to Path
        saved_config["output_dir"] = output_dir

        config = ReasonerConfig(**saved_config)

        # Create instance (this initializes fresh components)
        reasoner = cls(config)

        # Restore iteration state
        reasoner.iteration = state["iteration"]
        reasoner.cluster_history = state["cluster_history"]
        # Handle both old "problem" and new "prompts" format
        if "prompts" in state:
            reasoner._prompts = state["prompts"]
        elif "problem" in state and state["problem"]:
            reasoner._prompts = [state["problem"]]
        else:
            reasoner._prompts = []

        # Restore metrics history
        metrics_path = output_dir / "metrics.jsonl"
        if metrics_path.exists():
            with open(metrics_path) as f:
                for line in f:
                    reasoner.metrics_history.append(
                        IterationMetrics(**json.loads(line))
                    )

        # Restore vector_db
        vector_db_path = output_dir / "vector_db"
        if vector_db_path.exists():
            reasoner.vector_db = VectorDB.load(
                vector_db_path,
                active_pool_size=config.active_pool_size
            )
            # Reconnect attractor detector to loaded VectorDB
            reasoner.attractor_detector = AttractorDetector(
                vector_db=reasoner.vector_db,
                min_cluster_size=config.min_cluster_size
            )

        return reasoner

    def run(self, prompts: list[str] | str) -> ReasoningResult:
        """
        Run reasoning loop from fresh start. Terminates on dynamics, not protocols.

        Args:
            prompts: External prompt(s) to seed the pool (string or list of strings)

        Returns:
            ReasoningResult with pool state and trajectory
        """
        self._reset()

        # Normalize to list
        if isinstance(prompts, str):
            prompts = [prompts]

        self._prompts = list(prompts)

        # Seed pool with prompts
        for prompt in prompts:
            self._add_thought(f"{EXTERNAL_PROMPT_PREFIX}{prompt}", iteration=-1)

        if self.config.verbose:
            print(f"Prompts: {len(prompts)}")
            for p in prompts:
                preview = p[:60] + "..." if len(p) > 60 else p
                print(f"  >>> {preview}")
            print(f"Max iterations: {self.config.max_iterations}")
            print("-" * 40)

        return self._run_loop()

    # Alias for backwards compatibility
    solve = run

    def continue_run(self, prompts: list[str] | str | None = None) -> ReasoningResult:
        """
        Continue reasoning from current state (for resume).

        Use this after loading from checkpoint to continue iterations.
        Optionally inject new prompts before continuing.

        Args:
            prompts: Optional new prompt(s) to inject before continuing

        Returns:
            ReasoningResult with pool state and trajectory
        """
        # Inject new prompts if provided
        if prompts:
            if isinstance(prompts, str):
                prompts = [prompts]
            for prompt in prompts:
                self._add_thought(f"{EXTERNAL_PROMPT_PREFIX}{prompt}", self.iteration)
                self._prompts.append(prompt)

        if self.config.verbose:
            if self._prompts:
                print(f"Prompts: {len(self._prompts)}")
                for p in self._prompts[-3:]:  # Show last 3
                    preview = p[:60] + "..." if len(p) > 60 else p
                    print(f"  >>> {preview}")
                if len(self._prompts) > 3:
                    print(f"  ... and {len(self._prompts) - 3} more")
            print(f"Current iteration: {self.iteration}")
            print(f"Max iterations: {self.config.max_iterations}")
            print(f"Pool size: {self.vector_db.size()}")
            print("-" * 40)

        return self._run_loop()

    # Alias for backwards compatibility
    continue_solving = continue_run

    def _run_loop(self) -> ReasoningResult:
        """
        Core reasoning loop. Called by both solve() and continue_solving().

        Returns:
            ReasoningResult with pool state and trajectory
        """
        termination_reason = "timeout"

        while self.iteration < self.config.max_iterations:
            # Sample thoughts from working memory
            thoughts = self.vector_db.sample_random(
                self.config.k_samples,
                from_active_pool=True
            )

            if self.config.verbose:
                print(f"\n[Iteration {self.iteration}] Sampled {len(thoughts)} thoughts")

            # Invoke Mind (no protocol expectations)
            result = invoke_mind(
                system_prompt=self.config.system_prompt,
                messages=thoughts,
                token_limit=self.config.token_limit,
                model=self.config.model
            )

            # Store thinking (not transmitted)
            if result['thinking']:
                self.thinking_trace.append(result['thinking'])

            # Process transmitted thoughts
            thoughts_added = 0
            for thought in result['transmitted']:
                thought = thought.strip()
                if not thought:
                    continue
                self._add_thought(thought, self.iteration)
                thoughts_added += 1
                if self.config.verbose:
                    preview = thought[:60] + "..." if len(thought) > 60 else thought
                    print(f"  -> {preview}")

            # Measure dynamics
            metrics = self._measure_dynamics(thoughts_added)
            self.metrics_history.append(metrics)

            # Checkpoint after each iteration
            self._save_checkpoint()

            if self.config.verbose:
                print(f"  [metrics] clusters={metrics.num_clusters}, "
                      f"coherence={metrics.coherence:.2f}, "
                      f"diversity={metrics.diversity:.2f}")

            # Check termination (dynamics only)
            term = self._check_termination(metrics)
            if term:
                termination_reason = term
                if self.config.verbose:
                    print(f"\n[TERMINATED: {term}]")
                self.iteration += 1  # Mark this iteration as complete before saving
                break

            self.iteration += 1

        # Extract output from pool state
        dominant_texts = self._extract_dominant_cluster()
        final_state = self.attractor_detector.detect(self.iteration)

        # Final checkpoint
        self._save_checkpoint()

        return ReasoningResult(
            termination_reason=termination_reason,
            iterations=self.iteration + 1,
            total_thoughts=self.vector_db.size(),
            final_pool_size=self.vector_db.active_size(),
            dominant_cluster_texts=dominant_texts,
            cluster_count=final_state.get('num_clusters', 0),
            final_coherence=self._get_dominant_coherence(final_state),
            metrics_history=self.metrics_history,
            thinking_trace=self.thinking_trace,
        )

    def _reset(self):
        """Reset state for new session."""
        self.vector_db = VectorDB(
            active_pool_size=self.config.active_pool_size
        )
        self.attractor_detector = AttractorDetector(
            vector_db=self.vector_db,
            min_cluster_size=self.config.min_cluster_size
        )
        self.iteration = 0
        self.thinking_trace = []
        self.metrics_history = []
        self.cluster_history = []
        self._prompts = []

    def _save_checkpoint(self):
        """Save full state for resume capability."""
        if not self.config.output_dir:
            return

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # state.json - core counters and config
        # Convert config to dict, handling Path serialization
        config_dict = asdict(self.config)
        if config_dict.get('output_dir'):
            config_dict['output_dir'] = str(config_dict['output_dir'])

        state = {
            "iteration": self.iteration,
            "cluster_history": self.cluster_history,
            "config": config_dict,
            "prompts": self._prompts,
        }
        (self.config.output_dir / "state.json").write_text(
            json.dumps(state, indent=2)
        )

        # metrics.jsonl - full history
        # Convert numpy types to Python native for JSON serialization
        def to_native(obj):
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            return obj

        with open(self.config.output_dir / "metrics.jsonl", "w") as f:
            for m in self.metrics_history:
                f.write(json.dumps(to_native(asdict(m))) + "\n")

        # Save vector_db
        self.vector_db.save(self.config.output_dir / "vector_db")

    def _add_thought(self, thought: str, iteration: int):
        """Add thought to VectorDB with embedding."""
        embedding = self.embedding_client.embed_single(thought)
        if embedding is not None:
            self.vector_db.add(
                text=thought,
                embedding=embedding,
                round_num=iteration,
                mind_id=0
            )

    def _measure_dynamics(self, thoughts_added: int) -> IterationMetrics:
        """Measure pool dynamics for this iteration."""
        # Get attractor state
        attractor_state = self.attractor_detector.detect(self.iteration)
        clusters = attractor_state.get('clusters', [])

        # Dominant cluster stats
        if clusters:
            dominant = max(clusters, key=lambda c: c['size'])
            dominant_size = dominant['size']
            coherence = dominant['coherence']
        else:
            dominant_size = 0
            coherence = 0.0

        active_size = self.vector_db.active_size()
        dominant_share = dominant_size / active_size if active_size > 0 else 0.0

        # Diversity: mean pairwise distance in active pool
        diversity = self._compute_diversity()

        # Track cluster count for stability detection
        self.cluster_history.append(len(clusters))

        return IterationMetrics(
            iteration=self.iteration,
            pool_size=self.vector_db.size(),
            active_size=active_size,
            num_clusters=len(clusters),
            dominant_cluster_size=dominant_size,
            dominant_cluster_share=dominant_share,
            coherence=coherence,
            diversity=diversity,
            thoughts_added=thoughts_added,
        )

    def _compute_diversity(self) -> float:
        """Compute mean pairwise cosine distance in active pool."""
        embeddings, _ = self.vector_db.get_active_pool_data()
        if len(embeddings) < 2:
            return 0.0

        # Sample if too large (for efficiency)
        if len(embeddings) > 50:
            indices = np.random.choice(len(embeddings), 50, replace=False)
            embeddings = embeddings[indices]

        # Compute pairwise cosine distances
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)

        # Cosine similarity matrix
        sim_matrix = normalized @ normalized.T

        # Mean of upper triangle (excluding diagonal)
        n = len(sim_matrix)
        upper_indices = np.triu_indices(n, k=1)
        mean_sim = sim_matrix[upper_indices].mean()

        # Convert to distance
        return 1.0 - mean_sim

    def _check_termination(self, metrics: IterationMetrics) -> Optional[str]:
        """Check dynamics-based termination conditions."""

        # Skip early termination if disabled
        if self.config.no_early_termination:
            return None

        # Convergence: single dominant cluster with high coherence
        if (metrics.dominant_cluster_share > self.config.convergence_coverage
                and metrics.coherence > self.config.convergence_threshold):
            return "converged"

        # Stability: cluster structure unchanged for N iterations
        if len(self.cluster_history) >= self.config.stability_window:
            recent = self.cluster_history[-self.config.stability_window:]
            if len(set(recent)) == 1 and recent[0] > 0:
                # Same non-zero cluster count for stability_window iterations
                return "stable"

        return None

    def _extract_dominant_cluster(self) -> list[str]:
        """Extract representative texts from dominant cluster."""
        attractor_state = self.attractor_detector.detect(self.iteration)
        clusters = attractor_state.get('clusters', [])

        if not clusters:
            # No clusters - return recent messages
            _, metadata = self.vector_db.get_active_pool_data()
            return [m['text'] for m in metadata[-3:]]

        dominant = max(clusters, key=lambda c: c['size'])

        # Get representative messages
        if 'representative_ids' in dominant and dominant['representative_ids']:
            texts = []
            for vid in dominant['representative_ids'][:3]:
                meta = self.vector_db.get_message(vid)
                if meta:
                    texts.append(meta['text'])
            return texts

        return []

    def _get_dominant_coherence(self, attractor_state: dict) -> float:
        """Get coherence of dominant cluster."""
        clusters = attractor_state.get('clusters', [])
        if not clusters:
            return 0.0
        dominant = max(clusters, key=lambda c: c['size'])
        return dominant.get('coherence', 0.0)
