"""
Working Memory Reasoner - Pool-based thought ecology for reasoning.

Core insight: Reasoning as memetic selection. Thoughts compete for attention
in a working memory pool. Persistence requires transmission. The answer
emerges from semantic convergence, not sequential accumulation.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.core.vector_db import VectorDB
from src.core.embedding_client import EmbeddingClient
from src.core.mind import invoke_mind
from src.analysis.attractors import AttractorDetector


# Default system prompt for reasoning
REASONER_SYSTEM_PROMPT = """You are reasoning through a problem. You receive thoughts from your working memory.

Read the thoughts. Think privately. Then transmit thoughts back to working memory.

Transmitted thoughts persist and may be sampled again. Thoughts not transmitted are forgotten.

Guidelines:
- Build on promising thoughts
- Discard dead ends (don't re-transmit them)
- When confident in an answer, transmit: [ANSWER] your answer here

Format: Thoughts separated by --- on its own line.
Private thinking comes before the first ---."""


@dataclass
class ReasonerConfig:
    """Configuration for reasoning session."""

    # Pool parameters
    k_samples: int = 5  # Thoughts sampled per iteration
    active_pool_size: int = 50  # Working memory capacity

    # Termination
    max_iterations: int = 30
    convergence_threshold: float = 0.75  # Coherence for convergence detection
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
class ReasoningResult:
    """Result of a reasoning session."""
    answer: Optional[str]
    iterations: int
    termination_reason: str  # 'answer_tag', 'convergence', 'max_iterations'
    total_thoughts: int
    final_pool_size: int

    # Optional detailed data
    thinking_trace: list[str] = field(default_factory=list)
    convergence_history: list[dict] = field(default_factory=list)


class Reasoner:
    """
    Pool-based reasoner using VectorDB for working memory.

    The reasoning loop:
    1. Sample K thoughts from working memory (VectorDB)
    2. Invoke Mind with sampled thoughts
    3. Embed new thoughts, add to VectorDB
    4. Check termination (answer tag OR convergence OR max iterations)
    5. Repeat
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
        self.convergence_history: list[dict] = []

    def solve(self, problem: str) -> ReasoningResult:
        """
        Solve a problem using pool-based reasoning.

        Args:
            problem: The problem statement

        Returns:
            ReasoningResult with answer and metadata
        """
        self._reset()

        # Seed pool with problem
        self._add_thought(f"[PROBLEM] {problem}", iteration=-1)

        if self.config.verbose:
            print(f"Problem: {problem}")
            print(f"Max iterations: {self.config.max_iterations}")
            print("-" * 40)

        answer = None
        termination_reason = "max_iterations"

        while self.iteration < self.config.max_iterations:
            # Sample thoughts from working memory
            thoughts = self.vector_db.sample_random(
                self.config.k_samples,
                from_active_pool=True
            )

            if self.config.verbose:
                print(f"\n[Iteration {self.iteration}] Sampled {len(thoughts)} thoughts")

            # Invoke Mind
            result = invoke_mind(
                system_prompt=self.config.system_prompt,
                messages=thoughts,
                token_limit=self.config.token_limit
            )

            # Store thinking (not transmitted)
            if result['thinking']:
                self.thinking_trace.append(result['thinking'])

            # Process transmitted thoughts
            for thought in result['transmitted']:
                thought = thought.strip()
                if not thought:
                    continue

                # Check for explicit answer
                answer_match = self._extract_answer(thought)
                if answer_match:
                    answer = answer_match
                    termination_reason = "answer_tag"
                    self._add_thought(thought, self.iteration)
                    if self.config.verbose:
                        print(f"  [ANSWER FOUND] {answer}")
                    break

                # Add thought to pool
                self._add_thought(thought, self.iteration)
                if self.config.verbose:
                    preview = thought[:60] + "..." if len(thought) > 60 else thought
                    print(f"  → {preview}")

            if answer:
                break

            # Check for convergence
            if self.vector_db.active_size() >= self.config.min_cluster_size * 2:
                attractor_state = self.attractor_detector.detect(self.iteration)
                self.convergence_history.append(attractor_state)

                if self._check_convergence(attractor_state):
                    # Extract answer from dominant cluster
                    answer = self._extract_from_cluster(attractor_state)
                    termination_reason = "convergence"
                    if self.config.verbose:
                        print(f"\n[CONVERGENCE] Pool converged to attractor")
                        print(f"  Answer: {answer}")
                    break

            self.iteration += 1

        # Final extraction if no answer yet
        if not answer:
            answer = self._extract_final_answer()
            if self.config.verbose:
                print(f"\n[MAX ITERATIONS] Extracting best answer from pool")
                print(f"  Answer: {answer}")

        # Save if output_dir specified
        if self.config.output_dir:
            self.vector_db.save(self.config.output_dir / "vector_db")

        return ReasoningResult(
            answer=answer,
            iterations=self.iteration + 1,
            termination_reason=termination_reason,
            total_thoughts=self.vector_db.size(),
            final_pool_size=self.vector_db.active_size(),
            thinking_trace=self.thinking_trace,
            convergence_history=self.convergence_history
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
        self.convergence_history = []

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

    def _extract_answer(self, thought: str) -> Optional[str]:
        """Extract answer from [ANSWER] tagged thought."""
        match = re.search(r'\[ANSWER\]\s*(.+)', thought, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _check_convergence(self, attractor_state: dict) -> bool:
        """Check if pool has converged to a stable attractor."""
        if not attractor_state.get('detected'):
            return False

        clusters = attractor_state.get('clusters', [])
        if not clusters:
            return False

        # Check for dominant cluster with high coherence
        dominant = max(clusters, key=lambda c: c['size'])

        # Convergence: single dominant cluster OR very high coherence
        pool_coverage = dominant['size'] / self.vector_db.active_size()

        return (
            dominant['coherence'] > self.config.convergence_threshold
            and pool_coverage > 0.5
        )

    def _extract_from_cluster(self, attractor_state: dict) -> Optional[str]:
        """Extract answer from dominant cluster."""
        clusters = attractor_state.get('clusters', [])
        if not clusters:
            return None

        dominant = max(clusters, key=lambda c: c['size'])

        # Get representative message from cluster
        if 'representative_ids' in dominant and dominant['representative_ids']:
            rep_id = dominant['representative_ids'][0]
            meta = self.vector_db.get_message(rep_id)
            if meta:
                text = meta['text']
                # Try to extract answer tag first
                answer = self._extract_answer(text)
                if answer:
                    return answer
                # Otherwise return the representative text
                return text

        return None

    def _extract_final_answer(self) -> Optional[str]:
        """Extract best answer from pool when max iterations reached."""
        # Look for any [ANSWER] tagged messages in active pool
        _, metadata = self.vector_db.get_active_pool_data()

        for meta in reversed(metadata):  # Most recent first
            answer = self._extract_answer(meta['text'])
            if answer:
                return answer

        # No explicit answer - return most recent non-problem thought
        for meta in reversed(metadata):
            text = meta['text']
            if not text.startswith('[PROBLEM]'):
                return text

        return None
