"""
Working Memory Reasoner - Pool-based thought ecology.

Core philosophy: No explicit protocols. The pool state IS the output.

- No [ANSWER] tags - memes win by replication, not declaration
- Termination is dynamics-based (convergence, stability, timeout)
- We measure and steer the pool, not the answers
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.core.vector_db import VectorDB
from src.core.embedding_client import EmbeddingClient
from src.core.mind import invoke_mind
from src.analysis.attractors import AttractorDetector


# Minimal system prompt - no protocols, just framing
REASONER_SYSTEM_PROMPT = """You receive thoughts from a shared pool.

Read them. Think privately. Transmit thoughts worth keeping.

Transmitted thoughts persist and compete for attention.
Thoughts not transmitted are forgotten.

Format: Thoughts separated by --- on its own line."""


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

    # LLM
    model: str = "anthropic/claude-sonnet-4-20250514"
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

    def solve(self, problem: str) -> ReasoningResult:
        """
        Run reasoning loop. Terminates on dynamics, not protocols.

        Args:
            problem: The problem statement (seeded into pool)

        Returns:
            ReasoningResult with pool state and trajectory
        """
        self._reset()

        # Seed pool with problem
        self._add_thought(f"Problem: {problem}", iteration=-1)

        if self.config.verbose:
            print(f"Problem: {problem}")
            print(f"Max iterations: {self.config.max_iterations}")
            print("-" * 40)

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
                token_limit=self.config.token_limit
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
                break

            self.iteration += 1

        # Extract output from pool state
        dominant_texts = self._extract_dominant_cluster()
        final_state = self.attractor_detector.detect(self.iteration)

        # Save if output_dir specified
        if self.config.output_dir:
            self.vector_db.save(self.config.output_dir / "vector_db")

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
        """Reset state for new problem."""
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
