"""
Orchestrator - coordinates rounds, Minds, and VectorDB for Logosphere experiments.

Uses VectorDB for message storage with real-time embedding generation.
Supports attractor detection and intervention hooks.
"""

import time
from pathlib import Path
from typing import Optional

from .vector_db import VectorDB
from .logger import Logger
from .mind import invoke_mind
from .embedding_client import EmbeddingClient, EmbeddingAPIError
from .interventions import Intervention, create_intervention
from .. import config

# Import AttractorDetector lazily to avoid hdbscan dependency in core
try:
    from ..analysis.attractors import AttractorDetector
    ATTRACTOR_AVAILABLE = True
except ImportError:
    ATTRACTOR_AVAILABLE = False
    AttractorDetector = None


class ExperimentAbortError(Exception):
    """Raised when experiment must abort (e.g., embedding API failure)."""
    pass


class Orchestrator:
    def __init__(
        self,
        vector_db: VectorDB,
        logger: Logger,
        embedding_client: EmbeddingClient,
        n_minds: int = None,
        k_samples: int = None,
        system_prompt: str = None,
        token_limit: int = None,
        attractor_detector: "AttractorDetector" = None,
        intervention: Intervention = None,
        output_dir: Path = None,
    ):
        """
        Initialize orchestrator.

        Args:
            vector_db: VectorDB for message storage
            logger: Experiment logger
            embedding_client: Client for generating embeddings
            n_minds: Number of minds per round (defaults to config.N_MINDS)
            k_samples: Samples per mind (defaults to config.K_SAMPLES)
            system_prompt: System prompt (defaults to config.SYSTEM_PROMPT)
            token_limit: Token limit per mind (defaults to config.TOKEN_LIMIT)
            attractor_detector: Optional attractor detector for clustering
            intervention: Optional intervention for custom sampling
            output_dir: Directory for saving VectorDB (optional)
        """
        self.vector_db = vector_db
        self.logger = logger
        self.embedding_client = embedding_client
        self.n_minds = n_minds if n_minds is not None else config.N_MINDS
        self.k_samples = k_samples if k_samples is not None else config.K_SAMPLES
        self.system_prompt = system_prompt if system_prompt is not None else config.SYSTEM_PROMPT
        self.token_limit = token_limit if token_limit is not None else config.TOKEN_LIMIT
        self.attractor_detector = attractor_detector
        self.intervention = intervention
        self.output_dir = output_dir
        self.total_tokens = 0

        # Track last attractor state for intervention hooks
        self._attractor_state: Optional[dict] = None

    def run_round(self, round_num: int) -> int:
        """
        Execute one round - invoke all Minds sequentially.

        Args:
            round_num: Current round number (1-indexed)

        Returns:
            Number of messages added this round

        Raises:
            ExperimentAbortError: On embedding API failure
        """
        # Collect messages from all minds this round
        round_messages = []  # (text, mind_id) pairs

        # Log round start
        self.logger.log_round_start(
            round_num=round_num,
            pool_size=self.vector_db.size(),
            active_pool_size=self.vector_db.active_size()
        )

        # Pre-round intervention hook
        if self.intervention:
            self.intervention.on_round_start(round_num, self._attractor_state)

        # Invoke each Mind sequentially
        for mind_id in range(self.n_minds):
            # Sample K messages via intervention or direct
            if self.intervention:
                sampled_messages = self.intervention.on_sample(
                    k=self.k_samples,
                    round_num=round_num
                )
            else:
                sampled_messages = self.vector_db.sample_random(
                    self.k_samples,
                    from_active_pool=True
                )

            # Invoke Mind (fail fast on errors)
            result = invoke_mind(
                system_prompt=self.system_prompt,
                messages=sampled_messages,
                token_limit=self.token_limit
            )

            # Track tokens
            self.total_tokens += result['tokens_used']

            # Log Mind invocation (full content for analysis)
            self.logger.log_mind_invocation(
                round_num=round_num,
                mind_id=mind_id,
                input_messages=sampled_messages,
                thinking=result['thinking'],
                transmitted=result['transmitted'],
                tokens_used=result['tokens_used'],
                raw_output=result['raw_output']
            )

            # Collect transmitted messages for batch embedding
            for msg in result['transmitted']:
                round_messages.append((msg, mind_id))

        # Batch embed all transmitted messages
        vector_ids = []
        if round_messages:
            message_texts = [msg for msg, _ in round_messages]

            try:
                start_time = time.time()
                embeddings = self.embedding_client.embed_batch(message_texts)
                latency_ms = (time.time() - start_time) * 1000

                # Log embedding batch
                self.logger.log_embedding_batch(
                    round_num=round_num,
                    num_texts=len(message_texts),
                    latency_ms=latency_ms,
                    model=self.embedding_client.model,
                )

                # Add to VectorDB with embeddings
                for (msg, mind_id), embedding in zip(round_messages, embeddings):
                    vid = self.vector_db.add(
                        text=msg,
                        embedding=embedding,
                        round_num=round_num,
                        mind_id=mind_id
                    )
                    vector_ids.append(vid)

            except EmbeddingAPIError as e:
                self.logger.log_error(
                    f"Embedding failed: {e}",
                    round_num=round_num,
                    error_type="abort"
                )
                raise ExperimentAbortError(f"Round {round_num}: {e}") from e

        # Detect attractors (if enabled and enough data)
        if self.attractor_detector and self.vector_db.active_size() >= 5:
            self._attractor_state = self.attractor_detector.detect(round_num)
            self.logger.log_attractor_state(
                round_num=round_num,
                num_clusters=self._attractor_state['num_clusters'],
                noise_count=self._attractor_state['noise_count'],
                clusters=self._attractor_state['clusters'],
                detected=self._attractor_state['detected'],
            )

        # Post-round intervention hook
        if self.intervention:
            self.intervention.on_round_end(round_num, self._attractor_state, vector_ids)

        # Log round end
        self.logger.log_round_end(
            round_num=round_num,
            messages_added=len(vector_ids),
            vector_ids=vector_ids
        )

        # Save VectorDB periodically
        if self.output_dir and round_num % 10 == 0:
            self.vector_db.save(self.output_dir / "vector_db")

        return len(vector_ids)

    def run(self, max_rounds: int) -> None:
        """
        Execute full experiment.

        Args:
            max_rounds: Number of rounds to run

        Raises:
            ExperimentAbortError: On embedding API failure
        """
        for round_num in range(1, max_rounds + 1):
            self.run_round(round_num)

    def save(self) -> None:
        """Save VectorDB to output_dir."""
        if self.output_dir:
            self.vector_db.save(self.output_dir / "vector_db")
