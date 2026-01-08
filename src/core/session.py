"""
Session management for Logosphere.

Coordinates VectorDB, SnapshotStore, and InterventionLog for tracked operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable

from .vector_db import VectorDB
from .snapshots import SnapshotStore, Snapshot
from .intervention_log import (
    InterventionLog,
    Intervention,
    INTERVENTION_INJECT,
    INTERVENTION_FORK,
    INTERVENTION_ROLLBACK,
    INTERVENTION_RUN,
    INTERVENTION_SAVE,
)
from .embedding_client import EmbeddingClient


@dataclass
class SessionState:
    """Serializable session state (for persistence between CLI calls)."""

    current_snapshot_id: Optional[str]
    iteration: int
    config: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        return cls(**data)


class Session:
    """
    Manages a Logosphere session with full audit trail.

    All mutations to the pool are tracked as interventions with
    before/after snapshots for rollback and comparison.
    """

    def __init__(
        self,
        session_dir: Path,
        active_pool_size: int = 50,
        embedding_dim: int = 1536,
    ):
        """
        Initialize or open a session.

        Args:
            session_dir: Directory for session data
            active_pool_size: Active pool size for VectorDB
            embedding_dim: Embedding dimension
        """
        self.session_dir = Path(session_dir)
        self.active_pool_size = active_pool_size
        self.embedding_dim = embedding_dim

        # Components
        self.snapshot_store = SnapshotStore(self.session_dir / "snapshots")
        self.intervention_log = InterventionLog(self.session_dir / "interventions.jsonl")

        # State
        self._state_path = self.session_dir / "state.json"
        self.vector_db: Optional[VectorDB] = None
        self.current_snapshot_id: Optional[str] = None
        self.iteration: int = 0
        self.config: dict = {}

        # Load existing state or initialize
        if self._state_path.exists():
            self._load_state()
        else:
            self._init_state()

    def _load_state(self) -> None:
        """Load state from disk."""
        state = SessionState.from_dict(
            json.loads(self._state_path.read_text())
        )
        self.current_snapshot_id = state.current_snapshot_id
        self.iteration = state.iteration
        self.config = state.config

        # Load VectorDB from current snapshot
        if self.current_snapshot_id:
            self.vector_db = self.snapshot_store.load(
                self.current_snapshot_id,
                active_pool_size=self.active_pool_size
            )
        else:
            self.vector_db = VectorDB(
                active_pool_size=self.active_pool_size,
                embedding_dim=self.embedding_dim
            )

    def _init_state(self) -> None:
        """Initialize fresh state."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db = VectorDB(
            active_pool_size=self.active_pool_size,
            embedding_dim=self.embedding_dim
        )
        self.current_snapshot_id = None
        self.iteration = 0
        self.config = {}
        self._save_state()

    def _save_state(self) -> None:
        """Save state to disk."""
        state = SessionState(
            current_snapshot_id=self.current_snapshot_id,
            iteration=self.iteration,
            config=self.config,
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(state.to_dict(), indent=2))

    def save(
        self,
        description: str,
        metrics_fn: Optional[Callable[["VectorDB"], dict]] = None,
    ) -> Snapshot:
        """
        Manual save with annotation.

        Args:
            description: Human-readable description
            metrics_fn: Optional function to compute metrics from VectorDB

        Returns:
            Created Snapshot
        """
        metrics = metrics_fn(self.vector_db) if metrics_fn else {}

        snapshot = self.snapshot_store.save(
            vector_db=self.vector_db,
            description=description,
            iteration=self.iteration,
            parent_id=self.current_snapshot_id,
            config=self.config,
            metrics=metrics,
        )

        # Log as intervention
        self.intervention_log.record(
            intervention_type=INTERVENTION_SAVE,
            content={"description": description},
            snapshot_before=self.current_snapshot_id,
            snapshot_after=snapshot.id,
        )

        self.current_snapshot_id = snapshot.id
        self._save_state()

        return snapshot

    def load(self, snapshot_id: str) -> None:
        """
        Load snapshot (rollback), logged as intervention.

        Args:
            snapshot_id: Snapshot to load
        """
        snapshot = self.snapshot_store.get(snapshot_id)
        if snapshot is None:
            raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")

        old_snapshot_id = self.current_snapshot_id

        # Load VectorDB
        self.vector_db = self.snapshot_store.load(
            snapshot_id,
            active_pool_size=self.active_pool_size
        )
        self.iteration = snapshot.iteration
        self.config = snapshot.config
        self.current_snapshot_id = snapshot_id

        # Log rollback
        self.intervention_log.record(
            intervention_type=INTERVENTION_ROLLBACK,
            content={"target_snapshot": snapshot_id},
            snapshot_before=old_snapshot_id,
            snapshot_after=snapshot_id,
        )

        self._save_state()

    def fork(
        self,
        description: str,
        metrics_fn: Optional[Callable[["VectorDB"], dict]] = None,
    ) -> str:
        """
        Save fork point, return snapshot ID for later reference.

        Args:
            description: Fork description
            metrics_fn: Optional function to compute metrics

        Returns:
            Snapshot ID of fork point
        """
        metrics = metrics_fn(self.vector_db) if metrics_fn else {}

        snapshot = self.snapshot_store.save(
            vector_db=self.vector_db,
            description=f"fork: {description}",
            iteration=self.iteration,
            parent_id=self.current_snapshot_id,
            config=self.config,
            metrics=metrics,
        )

        # Log fork (current state unchanged, just marking a point)
        self.intervention_log.record(
            intervention_type=INTERVENTION_FORK,
            content={"description": description, "fork_snapshot": snapshot.id},
            snapshot_before=self.current_snapshot_id,
            snapshot_after=snapshot.id,
        )

        # Update current to fork point
        self.current_snapshot_id = snapshot.id
        self._save_state()

        return snapshot.id

    def inject_message(
        self,
        text: str,
        embedding_client: EmbeddingClient,
        notes: str = "",
        metrics_fn: Optional[Callable[["VectorDB"], dict]] = None,
    ) -> Intervention:
        """
        Add message to pool with full tracking.

        Auto-snapshots before and after injection.

        Args:
            text: Message text to inject
            embedding_client: Client for generating embedding
            notes: Optional observer notes
            metrics_fn: Optional function to compute metrics

        Returns:
            Intervention record
        """
        # Snapshot before
        before_metrics = metrics_fn(self.vector_db) if metrics_fn else {}
        before_snapshot = self.snapshot_store.save(
            vector_db=self.vector_db,
            description=f"pre-inject",
            iteration=self.iteration,
            parent_id=self.current_snapshot_id,
            config=self.config,
            metrics=before_metrics,
        )

        # Generate embedding and add
        embedding = embedding_client.embed_single(text)
        if embedding is None:
            raise ValueError("Embedding generation failed (client disabled?)")

        vector_id = self.vector_db.add(
            text=text,
            embedding=embedding,
            round_num=self.iteration,
            mind_id=-1,  # Sentinel for external injection
            extra_metadata={"injected": True},
        )

        # Snapshot after
        after_metrics = metrics_fn(self.vector_db) if metrics_fn else {}
        after_snapshot = self.snapshot_store.save(
            vector_db=self.vector_db,
            description=f"post-inject",
            iteration=self.iteration,
            parent_id=before_snapshot.id,
            config=self.config,
            metrics=after_metrics,
        )

        # Log intervention
        intervention = self.intervention_log.record(
            intervention_type=INTERVENTION_INJECT,
            content={
                "text": text,
                "vector_id": vector_id,
            },
            snapshot_before=before_snapshot.id,
            snapshot_after=after_snapshot.id,
            notes=notes,
        )

        self.current_snapshot_id = after_snapshot.id
        self._save_state()

        return intervention

    def record_run(
        self,
        iterations_run: int,
        snapshot_before_id: str,
        metrics_fn: Optional[Callable[["VectorDB"], dict]] = None,
        notes: str = "",
    ) -> Intervention:
        """
        Record a run intervention after iterations complete.

        Called by LogosRunner after batch execution.

        Args:
            iterations_run: Number of iterations run
            snapshot_before_id: Snapshot ID before run started
            metrics_fn: Optional function to compute metrics
            notes: Optional notes

        Returns:
            Intervention record
        """
        # Snapshot after
        after_metrics = metrics_fn(self.vector_db) if metrics_fn else {}
        after_snapshot = self.snapshot_store.save(
            vector_db=self.vector_db,
            description=f"post-run-{iterations_run}",
            iteration=self.iteration,
            parent_id=snapshot_before_id,
            config=self.config,
            metrics=after_metrics,
        )

        # Log intervention
        intervention = self.intervention_log.record(
            intervention_type=INTERVENTION_RUN,
            content={
                "iterations": iterations_run,
                "start_iteration": self.iteration - iterations_run,
                "end_iteration": self.iteration,
            },
            snapshot_before=snapshot_before_id,
            snapshot_after=after_snapshot.id,
            notes=notes,
        )

        self.current_snapshot_id = after_snapshot.id
        self._save_state()

        return intervention

    def prepare_run(
        self,
        metrics_fn: Optional[Callable[["VectorDB"], dict]] = None,
    ) -> str:
        """
        Prepare for a run by taking a pre-run snapshot.

        Returns:
            Snapshot ID to pass to record_run() after completion
        """
        metrics = metrics_fn(self.vector_db) if metrics_fn else {}
        snapshot = self.snapshot_store.save(
            vector_db=self.vector_db,
            description=f"pre-run",
            iteration=self.iteration,
            parent_id=self.current_snapshot_id,
            config=self.config,
            metrics=metrics,
        )
        self.current_snapshot_id = snapshot.id
        self._save_state()
        return snapshot.id

    def get_status(self) -> dict:
        """
        Get current session status.

        Returns:
            Dict with pool size, iteration, snapshot info
        """
        return {
            "session_dir": str(self.session_dir),
            "iteration": self.iteration,
            "current_snapshot_id": self.current_snapshot_id,
            "pool_size": self.vector_db.size() if self.vector_db else 0,
            "active_pool_size": self.vector_db.active_size() if self.vector_db else 0,
            "total_snapshots": len(self.snapshot_store.list()),
            "total_interventions": len(self.intervention_log.all()),
        }
