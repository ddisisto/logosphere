"""
Logos reasoning runner.

Core loop: sample → mind → embed → add.
Integrates with Session for snapshot/intervention tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from src.core.session import Session
from src.core.intervention_log import Intervention


def _to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


from src.core.vector_db import VectorDB
from src.core.embedding_client import EmbeddingClient
from src.core.mind import invoke_mind
from src.analysis.attractors import AttractorDetector

from .config import LogosConfig, EXTERNAL_PROMPT_PREFIX


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
    diversity: float
    thoughts_added: int


def compute_metrics(vector_db: VectorDB, min_cluster_size: int = 3) -> dict:
    """
    Compute pool metrics for snapshots.

    Args:
        vector_db: VectorDB to analyze
        min_cluster_size: Minimum cluster size for HDBSCAN

    Returns:
        Dict with diversity, cluster info
    """
    if vector_db.active_size() < min_cluster_size:
        return {
            "diversity": 0.0,
            "num_clusters": 0,
            "dominant_cluster_size": 0,
            "dominant_cluster_share": 0.0,
            "coherence": 0.0,
        }

    try:
        detector = AttractorDetector(vector_db, min_cluster_size=min_cluster_size)
        state = detector.detect(round_num=-1)  # -1 for snapshot context

        clusters = state.get('clusters', [])
        if clusters:
            dominant = max(clusters, key=lambda c: c['size'])
            dominant_size = dominant['size']
            coherence = dominant['coherence']
        else:
            dominant_size = 0
            coherence = 0.0

        active_size = vector_db.active_size()
        dominant_share = dominant_size / active_size if active_size > 0 else 0.0

        # Diversity: mean pairwise distance
        diversity = _compute_diversity(vector_db)

        return _to_native({
            "diversity": diversity,
            "num_clusters": len(clusters),
            "dominant_cluster_size": dominant_size,
            "dominant_cluster_share": dominant_share,
            "coherence": coherence,
        })
    except ImportError:
        # HDBSCAN not available
        return _to_native({
            "diversity": _compute_diversity(vector_db),
            "num_clusters": 0,
            "dominant_cluster_size": 0,
            "dominant_cluster_share": 0.0,
            "coherence": 0.0,
        })


def _compute_diversity(vector_db: VectorDB) -> float:
    """Compute mean pairwise cosine distance in active pool."""
    embeddings, _ = vector_db.get_active_pool_data()
    if len(embeddings) < 2:
        return 0.0

    # Sample if too large
    if len(embeddings) > 50:
        indices = np.random.choice(len(embeddings), 50, replace=False)
        embeddings = embeddings[indices]

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Cosine similarity matrix
    sim_matrix = normalized @ normalized.T

    # Mean of upper triangle
    n = len(sim_matrix)
    upper_indices = np.triu_indices(n, k=1)
    mean_sim = sim_matrix[upper_indices].mean()

    return 1.0 - mean_sim


class LogosRunner:
    """
    Runs reasoning iterations with session integration.

    Supports both single-step and batch modes.
    """

    def __init__(self, session: Session, config: LogosConfig):
        """
        Initialize runner.

        Args:
            session: Session to use (provides VectorDB)
            config: Configuration
        """
        self.session = session
        self.config = config

        # Components
        self.embedding_client = EmbeddingClient(
            model=config.embedding_model,
            enabled=True
        )

        # Attractor detector (created on demand)
        self._attractor_detector: Optional[AttractorDetector] = None

        # Tracking
        self.thinking_trace: list[str] = []
        self.cluster_history: list[int] = []

    @property
    def attractor_detector(self) -> AttractorDetector:
        """Lazy attractor detector creation."""
        if self._attractor_detector is None:
            self._attractor_detector = AttractorDetector(
                vector_db=self.session.vector_db,
                min_cluster_size=self.config.min_cluster_size
            )
        return self._attractor_detector

    def seed_prompts(self, prompts: list[str]) -> None:
        """
        Seed pool with initial prompts.

        Args:
            prompts: List of prompt strings
        """
        for prompt in prompts:
            text = f"{EXTERNAL_PROMPT_PREFIX}{prompt}"
            embedding = self.embedding_client.embed_single(text)
            if embedding is not None:
                self.session.add(
                    text=text,
                    embedding=embedding,
                    mind_id=-1,
                    extra_metadata={"seed": True},
                )

    def step(self) -> IterationMetrics:
        """
        Execute single iteration: sample → mind → embed → add.

        Returns:
            Metrics for this iteration
        """
        # Sample from current branch's visible pool
        thoughts, sampled_ids = self.session.sample(self.config.k_samples)

        if self.config.verbose:
            print(f"\n[Iteration {self.session.iteration}] Sampled {len(thoughts)} thoughts")

        # Invoke Mind
        result = invoke_mind(
            system_prompt=self.config.system_prompt,
            messages=thoughts,
            token_limit=self.config.token_limit,
            model=self.config.model
        )

        # Store thinking
        if result['thinking']:
            self.thinking_trace.append(result['thinking'])

        # Process transmitted thoughts
        thoughts_added = 0
        for thought in result['transmitted']:
            thought = thought.strip()
            if not thought:
                continue

            embedding = self.embedding_client.embed_single(thought)
            if embedding is not None:
                self.session.add(
                    text=thought,
                    embedding=embedding,
                    mind_id=0,
                    extra_metadata={"sampled_ids": sampled_ids},
                )
                thoughts_added += 1

                if self.config.verbose:
                    preview = thought[:60] + "..." if len(thought) > 60 else thought
                    print(f"  -> {preview}")

        # Measure dynamics
        metrics = self._measure_dynamics(thoughts_added)

        if self.config.verbose:
            print(f"  [metrics] clusters={metrics.num_clusters}, "
                  f"coherence={metrics.coherence:.2f}, "
                  f"diversity={metrics.diversity:.2f}")

        # Increment iteration
        self.session.iteration += 1

        return metrics

    def run(self, iterations: int) -> list[IterationMetrics]:
        """
        Run batch iterations with session tracking.

        Args:
            iterations: Number of iterations to run

        Returns:
            List of metrics for each iteration
        """
        metrics_history = []
        termination_reason = "completed"

        for i in range(iterations):
            metrics = self.step()
            metrics_history.append(metrics)

            # Check termination (if any conditions are configured)
            term = self._check_termination(metrics)
            if term:
                termination_reason = term
                if self.config.verbose:
                    print(f"\n[TERMINATED: {term}]")
                break

        # Record run intervention
        self.session.record_run(
            iterations_run=len(metrics_history),
            notes=f"termination: {termination_reason}",
        )

        return metrics_history

    def _measure_dynamics(self, thoughts_added: int) -> IterationMetrics:
        """Measure pool dynamics for this iteration."""
        vector_db = self.session.vector_db

        try:
            attractor_state = self.attractor_detector.detect(self.session.iteration)
            clusters = attractor_state.get('clusters', [])

            if clusters:
                dominant = max(clusters, key=lambda c: c['size'])
                dominant_size = dominant['size']
                coherence = dominant['coherence']
            else:
                dominant_size = 0
                coherence = 0.0
        except (ImportError, Exception):
            clusters = []
            dominant_size = 0
            coherence = 0.0

        active_size = vector_db.active_size()
        dominant_share = dominant_size / active_size if active_size > 0 else 0.0

        diversity = _compute_diversity(vector_db)

        # Track cluster count for stability
        self.cluster_history.append(len(clusters))

        return IterationMetrics(
            iteration=self.session.iteration,
            pool_size=vector_db.size(),
            active_size=active_size,
            num_clusters=len(clusters),
            dominant_cluster_size=dominant_size,
            dominant_cluster_share=dominant_share,
            coherence=coherence,
            diversity=diversity,
            thoughts_added=thoughts_added,
        )

    def _check_termination(self, metrics: IterationMetrics) -> Optional[str]:
        """Check dynamics-based termination conditions (if configured)."""
        # Convergence (only if both thresholds are set)
        if (self.config.convergence_threshold is not None and
                self.config.convergence_coverage is not None):
            if (metrics.dominant_cluster_share > self.config.convergence_coverage
                    and metrics.coherence > self.config.convergence_threshold):
                return "converged"

        # Stability (only if window is set)
        if self.config.stability_window is not None:
            if len(self.cluster_history) >= self.config.stability_window:
                recent = self.cluster_history[-self.config.stability_window:]
                if len(set(recent)) == 1 and recent[0] > 0:
                    return "stable"

        return None
