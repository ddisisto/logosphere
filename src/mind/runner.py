"""
Mind v2 Runner - Core loop for dual-pool reasoning.

Sample thoughts → invoke mind → embed thoughts → cluster → add messages
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.core.session_v2 import SessionV2
from src.core.embedding_client import EmbeddingClient
from src.core.mind_v2 import (
    format_input,
    invoke_mind,
    load_system_prompt,
    MindOutput,
)
from src.logos.clustering import ClusterManager

from .config import MindConfig


@dataclass
class StepResult:
    """Result of a single iteration."""
    thoughts_added: int
    messages_added: int
    skipped: bool
    raw_output: str


class MindRunner:
    """
    Runner for Mind v2 protocol.

    Core loop:
    1. Sample thoughts from thinking_pool
    2. Get messages from message_pool
    3. Format YAML input
    4. Invoke Mind (LLM)
    5. Parse YAML output
    6. Embed new thoughts → thinking_pool
    7. Add new messages → message_pool
    8. Run incremental clustering on new thoughts
    """

    def __init__(
        self,
        session: SessionV2,
        config: Optional[MindConfig] = None,
    ):
        self.session = session
        self.config = config or MindConfig()

        # Embedding client (reuse from v1)
        self.embedding_client = EmbeddingClient(
            model=session.config.embedding_model,
            api_key=None,  # Will use env var
        )

        # Clustering manager
        self.cluster_mgr = ClusterManager(session.session_dir)

        # Load system prompt
        self.system_prompt = load_system_prompt()

        if self.config.verbose and self.cluster_mgr.initialized:
            print("[clustering] Incremental clustering enabled")

    def step(self) -> StepResult:
        """
        Execute single iteration.

        Returns:
            StepResult with counts and status
        """
        # 1. Sample thoughts from thinking pool
        thoughts, sampled_ids = self.session.sample_thoughts(self.config.k_samples)

        # 2. Get cluster assignments for sampled thoughts
        cluster_assignments = {}
        if self.cluster_mgr.initialized and self.cluster_mgr.assignments:
            for vid in sampled_ids:
                entry = self.cluster_mgr.assignments.get(vid)
                if entry:
                    cluster_assignments[vid] = {
                        'cluster_id': entry.cluster_id,
                        'noise_since': entry.noise_since,
                    }

        # 3. Get messages from message pool
        messages = self.session.get_messages()

        if self.config.verbose:
            print(f"\n[Iteration {self.session.iteration}] "
                  f"Sampled {len(thoughts)} thoughts, {len(messages)} messages")

        # 4. Format YAML input
        user_input = format_input(
            mind_id=self.config.mind_id,
            current_iter=self.session.iteration,
            thoughts=thoughts,
            messages=messages,
            cluster_assignments=cluster_assignments,
        )

        # 5. Invoke Mind
        output: MindOutput = invoke_mind(
            system_prompt=self.system_prompt,
            user_input=user_input,
            model=self.config.model,
            token_limit=self.config.token_limit,
        )

        if output.skipped:
            if self.config.verbose:
                print("  [skipped] Mind opted out this iteration")
            self.session.iteration += 1
            self.session.save()
            return StepResult(
                thoughts_added=0,
                messages_added=0,
                skipped=True,
                raw_output=output.raw,
            )

        # 6. Process new thoughts → thinking_pool
        thoughts_added = 0
        new_thought_ids = []
        for thought_text in output.thoughts:
            thought_text = thought_text.strip()
            if not thought_text:
                continue

            # Embed
            embedding = self.embedding_client.embed_single(thought_text)
            if embedding is not None:
                vid = self.session.add_thought(thought_text, embedding)
                new_thought_ids.append(vid)
                thoughts_added += 1

                if self.config.verbose:
                    preview = thought_text[:60] + "..." if len(thought_text) > 60 else thought_text
                    print(f"  [thought] {preview}")

        # 7. Process new messages → message_pool
        messages_added = 0
        for msg in output.messages:
            to = msg.get('to', 'user')
            text = msg.get('text', '').strip()
            if not text:
                continue

            self.session.add_message(
                source=self.config.mind_id,
                to=to,
                text=text,
            )
            messages_added += 1

            if self.config.verbose:
                preview = text[:60] + "..." if len(text) > 60 else text
                print(f"  [message → {to}] {preview}")

        # 8. Run incremental clustering on new thoughts
        if self.cluster_mgr.initialized and new_thought_ids:
            try:
                stats = self.cluster_mgr.process(
                    session=self._make_clustering_adapter(),
                    iteration=self.session.iteration,
                    min_cluster_size=self.session.config.min_cluster_size,
                    centroid_threshold=self.session.config.centroid_match_threshold,
                    noise_window=self.session.config.noise_reconsider_iterations,
                )
                if self.config.verbose:
                    print(f"  [clustering] assigned={stats.get('assigned', 0)}, "
                          f"new_clusters={stats.get('new_clusters', 0)}")

                # Update cluster assignments in thinking pool
                if self.cluster_mgr.assignments:
                    for vid in new_thought_ids:
                        entry = self.cluster_mgr.assignments.get(vid)
                        if entry:
                            self.session.thinking_pool.update_cluster(vid, entry.cluster_id)

            except Exception as e:
                if self.config.verbose:
                    print(f"  [clustering] error: {e}")

        # 9. Increment iteration and save
        self.session.iteration += 1
        self.session.save()

        return StepResult(
            thoughts_added=thoughts_added,
            messages_added=messages_added,
            skipped=False,
            raw_output=output.raw,
        )

    def run(self, iterations: int) -> list[StepResult]:
        """
        Run multiple iterations.

        Args:
            iterations: Number of iterations to run

        Returns:
            List of StepResults
        """
        results = []

        if self.config.verbose:
            print(f"Running {iterations} iterations...")
            print("-" * 40)

        for i in range(iterations):
            result = self.step()
            results.append(result)

        if self.config.verbose:
            print("-" * 40)
            total_thoughts = sum(r.thoughts_added for r in results)
            total_messages = sum(r.messages_added for r in results)
            skipped = sum(1 for r in results if r.skipped)
            print(f"Completed {iterations} iterations: "
                  f"{total_thoughts} thoughts, {total_messages} messages, "
                  f"{skipped} skipped")

        return results

    def bootstrap_clustering(self) -> None:
        """Bootstrap clustering from existing thoughts."""
        if self.session.thinking_pool.size() == 0:
            print("No thoughts to cluster")
            return

        self.cluster_mgr.bootstrap(
            session=self._make_clustering_adapter(),
            min_cluster_size=self.session.config.min_cluster_size,
        )

        # Update cluster assignments in thinking pool
        if self.cluster_mgr.assignments:
            for vid in range(self.session.thinking_pool.size()):
                entry = self.cluster_mgr.assignments.get(vid)
                if entry:
                    self.session.thinking_pool.update_cluster(vid, entry.cluster_id)

        self.session.thinking_pool.save()

        if self.config.verbose:
            status = self.cluster_mgr.get_status()
            print(f"Bootstrapped clustering: {status.get('num_clusters', 0)} clusters")

    def _make_clustering_adapter(self):
        """
        Create adapter to make ThinkingPool compatible with clustering API.

        The clustering code expects:
        - session.get_visible_ids() -> set[int]
        - session.vector_db.get_message(vid) -> dict with 'text', 'round'
        - session.vector_db.embeddings[vid] -> np.ndarray
        """
        return _ClusteringAdapter(self.session.thinking_pool)


class _ClusteringAdapter:
    """
    Adapter to make ThinkingPool compatible with existing clustering code.

    The clustering algorithm expects a session-like interface.
    """

    def __init__(self, thinking_pool):
        self.thinking_pool = thinking_pool
        # Create a fake vector_db interface
        self.vector_db = _VectorDBAdapter(thinking_pool)

    def get_visible_ids(self) -> set[int]:
        """Get all thought vector IDs."""
        return self.thinking_pool.get_visible_ids()


class _VectorDBAdapter:
    """Adapter to make ThinkingPool look like VectorDB for clustering."""

    def __init__(self, thinking_pool):
        self.thinking_pool = thinking_pool
        # Expose embeddings as indexable
        self._embeddings = None

    @property
    def embeddings(self):
        """Return embeddings list for indexing."""
        if self._embeddings is None:
            self._embeddings = [t.embedding for t in self.thinking_pool.thoughts]
        return self._embeddings

    def get_message(self, vid: int) -> Optional[dict]:
        """Get message dict for clustering compatibility."""
        return self.thinking_pool.get_message(vid)
