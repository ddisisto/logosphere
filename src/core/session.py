"""
Session management for Logosphere.

Linear session model: single append-only VectorDB, no branching.
Fork sessions by extracting/cloning with extract_session.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .vector_db import VectorDB
from .embedding_client import EmbeddingClient
from .intervention_log import (
    InterventionLog,
    Intervention,
    INTERVENTION_INJECT,
    INTERVENTION_RUN,
)

# External messages (injected/seeded) get this prefix
EXTERNAL_PROMPT_PREFIX = ">>> "


class Session:
    """
    Manages a Logosphere session.

    Single append-only VectorDB. Linear history.
    """

    def __init__(
        self,
        session_dir: Path,
        active_pool_size: int = 50,
        embedding_dim: int = 1536,
    ):
        self.session_dir = Path(session_dir)
        self.active_pool_size = active_pool_size
        self.embedding_dim = embedding_dim

        # Paths
        self._session_path = self.session_dir / "session.json"
        self._vector_db_path = self.session_dir / "vector_db"

        # State
        self.vector_db: Optional[VectorDB] = None
        self.iteration: int = 0
        self.config: dict = {}

        # Intervention log (optional, for audit)
        self.intervention_log = InterventionLog(self.session_dir / "interventions.jsonl")

        # Load or init
        if self._session_path.exists():
            self._load()
        else:
            self._init()

    def _init(self) -> None:
        """Initialize fresh session."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.vector_db = VectorDB(
            active_pool_size=self.active_pool_size,
            embedding_dim=self.embedding_dim
        )

        self.iteration = 0
        self.config = {}

        self._save()

    def _load(self) -> None:
        """Load session from disk."""
        with open(self._session_path) as f:
            data = json.load(f)

        self.iteration = data.get("iteration", 0)
        self.config = data.get("config", {})

        # Update active_pool_size from config if set
        if self.config.get('active_pool_size'):
            self.active_pool_size = self.config['active_pool_size']

        # Load VectorDB
        if self._vector_db_path.exists():
            self.vector_db = VectorDB.load(
                self._vector_db_path,
                active_pool_size=self.active_pool_size
            )
        else:
            self.vector_db = VectorDB(
                active_pool_size=self.active_pool_size,
                embedding_dim=self.embedding_dim
            )

    def _save(self) -> None:
        """Save session to disk."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "iteration": self.iteration,
            "config": self.config,
        }
        with open(self._session_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Save VectorDB
        self.vector_db.save(self._vector_db_path)

    def get_visible_ids(self) -> set[int]:
        """Get all vector_ids in the session."""
        return set(range(self.vector_db.size()))

    def sample(self, k: int) -> tuple[list[str], list[int]]:
        """
        Sample k messages from active pool.

        Returns:
            (texts, vector_ids)
        """
        total = self.vector_db.size()
        if total == 0:
            return [], []

        # Get active pool (tail of all messages)
        if total > self.active_pool_size:
            active_ids = list(range(total - self.active_pool_size, total))
        else:
            active_ids = list(range(total))

        # Sample
        import random
        sample_size = min(k, len(active_ids))
        sampled_ids = random.sample(active_ids, sample_size)

        texts = [self.vector_db.get_message(vid)['text'] for vid in sampled_ids]
        return texts, sampled_ids

    def add(
        self,
        text: str,
        embedding,
        mind_id: int = 0,
        extra_metadata: Optional[dict] = None,
    ) -> int:
        """Add message to VectorDB."""
        vector_id = self.vector_db.add(
            text=text,
            embedding=embedding,
            round_num=self.iteration,
            mind_id=mind_id,
            extra_metadata=extra_metadata,
        )
        return vector_id

    def inject_message(
        self,
        text: str,
        embedding_client: EmbeddingClient,
        notes: str = "",
    ) -> Intervention:
        """Add external message with intervention logging."""
        # Add external prompt prefix for consistency with seeded prompts
        prefixed_text = f"{EXTERNAL_PROMPT_PREFIX}{text}"

        embedding = embedding_client.embed_single(prefixed_text)
        if embedding is None:
            raise ValueError("Embedding generation failed")

        vector_id = self.add(
            text=prefixed_text,
            embedding=embedding,
            mind_id=-1,
            extra_metadata={"injected": True},
        )

        intervention = self.intervention_log.record(
            intervention_type=INTERVENTION_INJECT,
            content={"text": text, "vector_id": vector_id},
            snapshot_before=f"iter_{self.iteration}",
            snapshot_after=f"iter_{self.iteration}",
            notes=notes,
        )

        self._save()
        return intervention

    def record_run(self, iterations_run: int, notes: str = "") -> Intervention:
        """Record a run intervention."""
        intervention = self.intervention_log.record(
            intervention_type=INTERVENTION_RUN,
            content={
                "iterations": iterations_run,
                "end_iteration": self.iteration,
            },
            snapshot_before=f"iter_{self.iteration - iterations_run}",
            snapshot_after=f"iter_{self.iteration}",
            notes=notes,
        )
        self._save()
        return intervention

    def get_status(self) -> dict:
        """Get current session status."""
        total = self.vector_db.size() if self.vector_db else 0
        return {
            "session_dir": str(self.session_dir),
            "iteration": self.iteration,
            "total_messages": total,
            "active_pool_size": min(total, self.active_pool_size),
            "interventions": len(self.intervention_log.all()),
        }
