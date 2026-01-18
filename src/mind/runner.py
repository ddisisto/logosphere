"""
Mind v2 Runner - Core loop for dual-pool reasoning.

Sample thoughts → invoke mind → embed thoughts → cluster → add draft
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
    draft_added: bool
    hard_signal: bool  # Mind output no draft when drafts exist = attention demand
    skipped: bool
    raw_output: str


class MindRunner:
    """
    Runner for Mind v2 protocol.

    Core loop:
    1. Sample thoughts from thinking_pool
    2. Get dialogue state (awaiting/drafts/history)
    3. Format YAML input
    4. Invoke Mind (LLM)
    5. Parse YAML output
    6. Embed new thoughts → thinking_pool
    7. Add new draft → dialogue_pool (if any)
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

        # Clustering manager (auto-initializes on first use)
        self.cluster_mgr = ClusterManager(session.session_dir)

        # Load system prompt
        self.system_prompt = load_system_prompt()

    def _step_inner(self) -> StepResult:
        """
        Execute single iteration (internal implementation).

        Raises:
            RuntimeError: If not in drafting state (no awaiting message)

        Returns:
            StepResult with counts and status
        """
        # Guard: must be in drafting state
        if not self.session.is_drafting:
            raise RuntimeError(
                "Cannot run iterations while idle. "
                "Send a message first with 'mind message'."
            )

        # 1. Sample thoughts from thinking pool
        thoughts, sampled_ids = self.session.sample_thoughts(self.session.config.k_samples)

        # 2. Get cluster assignments and sizes for sampled thoughts
        cluster_assignments = {}
        if self.cluster_mgr.assignments:
            # Compute active cluster sizes (members currently in visible pool)
            visible_ids = self.session.thinking_pool.get_visible_ids()
            cluster_sizes = {}
            for vid in visible_ids:
                entry = self.cluster_mgr.assignments.get(vid)
                if entry and entry.cluster_id.startswith('cluster_'):
                    cluster_sizes[entry.cluster_id] = cluster_sizes.get(entry.cluster_id, 0) + 1

            # Build assignments with sizes
            for vid in sampled_ids:
                entry = self.cluster_mgr.assignments.get(vid)
                if entry:
                    size = cluster_sizes.get(entry.cluster_id) if entry.cluster_id.startswith('cluster_') else None
                    cluster_assignments[vid] = {'cluster_id': entry.cluster_id, 'size': size}

        if self.config.verbose:
            state = "drafting" if self.session.is_drafting else "idle"
            print(f"\n[Iteration {self.session.iteration}] "
                  f"Sampled {len(thoughts)} thoughts, state={state}")

        # 3. Get display-limited drafts and history
        drafts_for_display = self.session.get_drafts_for_mind()
        history_for_display = self.session.get_history_for_mind()

        # 4. Format YAML input with dialogue pool
        user_input = format_input(
            mind_id=self.config.mind_id,
            current_iter=self.session.iteration,
            thoughts=thoughts,
            dialogue_pool=self.session.dialogue_pool,
            drafts_for_display=drafts_for_display,
            history_for_display=history_for_display,
            cluster_assignments=cluster_assignments,
        )

        if self.config.debug:
            print("\n" + "=" * 60)
            print("DEBUG: LLM REQUEST")
            print("=" * 60)
            print(f"Model: {self.session.config.model}")
            print(f"Token limit: {self.session.config.token_limit}")
            print("-" * 60)
            print("USER INPUT:")
            print("-" * 60)
            print(user_input)
            print("=" * 60)

        # 4. Invoke Mind
        output: MindOutput = invoke_mind(
            system_prompt=self.system_prompt,
            user_input=user_input,
            model=self.session.config.model,
            token_limit=self.session.config.token_limit,
        )

        if self.config.debug:
            print("\n" + "=" * 60)
            print("DEBUG: LLM RESPONSE")
            print("=" * 60)
            print(output.raw)
            print("=" * 60 + "\n")

        if output.skipped:
            if self.config.verbose:
                print("  [skipped] Mind opted out this iteration")
            self.session.iteration += 1
            self.session.save()
            return StepResult(
                thoughts_added=0,
                draft_added=False,
                hard_signal=False,
                skipped=True,
                raw_output=output.raw,
            )

        # 5. Process new thoughts → thinking_pool
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

        # 6. Process draft → dialogue_pool
        draft_added = False
        hard_signal = False
        if output.draft:
            self.session.add_draft(output.draft)
            draft_added = True

            if self.config.verbose:
                preview = output.draft[:60] + "..." if len(output.draft) > 60 else output.draft
                print(f"  [draft] {preview}")
        else:
            # Hard signal: no draft output when drafting with existing drafts
            # Mind is saying "look at what's there"
            if self.session.is_drafting and self.session.dialogue_pool.drafts:
                hard_signal = True
                if self.config.verbose:
                    print(f"  [SIGNAL] hard stop - {len(self.session.dialogue_pool.drafts)} drafts ready")

        # 7. Run incremental clustering on new thoughts (auto-initializes if needed)
        if new_thought_ids:
            try:
                stats = self.cluster_mgr.process(
                    session=self._make_clustering_adapter(),
                    iteration=self.session.iteration,
                    min_cluster_size=self.session.config.min_cluster_size,
                    centroid_threshold=self.session.config.centroid_match_threshold,
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

        # 8. Increment iteration and save
        self.session.iteration += 1
        self.session.save()

        return StepResult(
            thoughts_added=thoughts_added,
            draft_added=draft_added,
            hard_signal=hard_signal,
            skipped=False,
            raw_output=output.raw,
        )

    def run(self, iterations: Optional[int] = None, max_iterations: int = 100) -> list[StepResult]:
        """
        Run iterations with automatic stop detection.

        Args:
            iterations: Exact count to run. If None, run until stop condition.
            max_iterations: Safety limit when iterations is None.

        Stop conditions (when iterations is None):
            - draft_added: New draft produced
            - hard_signal: Mind signaled attention needed (no draft when drafts exist)

        Returns:
            List of StepResults
        """
        results = []
        limit = iterations if iterations is not None else max_iterations

        if self.config.verbose:
            if iterations is not None:
                print(f"Running {iterations} iterations...")
            else:
                print("Running until stop condition...")
            print("-" * 40)

        for i in range(limit):
            result = self._step_inner()
            results.append(result)

            # Check stop conditions (only when running until condition)
            if iterations is None and self._should_stop(result):
                break

            # Exact count reached
            if iterations is not None and len(results) >= iterations:
                break

        if self.config.verbose:
            self._print_summary(results, iterations, max_iterations)

        return results

    def step(self) -> StepResult:
        """Single iteration. Equivalent to run(1)[0]."""
        return self.run(1)[0]

    def _should_stop(self, result: StepResult) -> bool:
        """Check if we should stop running (for run-until-condition mode)."""
        if result.draft_added:
            if self.config.verbose:
                print("  → stopping: draft produced")
            return True
        if result.hard_signal:
            if self.config.verbose:
                print("  → stopping: hard signal (attention demanded)")
            return True
        return False

    def _print_summary(self, results: list[StepResult], iterations: Optional[int], max_iterations: int):
        """Print run summary."""
        print("-" * 40)
        total_thoughts = sum(r.thoughts_added for r in results)
        total_drafts = sum(1 for r in results if r.draft_added)
        hard_signals = sum(1 for r in results if r.hard_signal)
        skipped = sum(1 for r in results if r.skipped)

        if iterations is not None:
            print(f"Completed {len(results)} iterations: "
                  f"{total_thoughts} thoughts, {total_drafts} drafts, "
                  f"{skipped} skipped")
        else:
            stop_reason = "draft" if total_drafts > 0 else "hard signal" if hard_signals > 0 else "max reached"
            print(f"Stopped after {len(results)} iterations ({stop_reason}): "
                  f"{total_thoughts} thoughts, {total_drafts} drafts")

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
