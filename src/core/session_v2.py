"""
Session v2 - Dual-pool session management for Logosphere v2.

Manages thinking_pool and dialogue_pool with YAML-based config.

Session structure:
    session/
    ├── session.yaml          # Config + iteration state
    ├── thinking/
    │   ├── embeddings.npy
    │   └── pool.yaml
    ├── dialogue/
    │   └── pool.yaml
    ├── clusters/             # Unchanged from v1
    └── interventions.jsonl   # Audit trail
"""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path
from typing import Optional

import yaml

from .thinking_pool import ThinkingPool, Thought
from .dialogue_pool import DialoguePool, Draft, HistoryEntry
from .intervention_log import InterventionLog


class SessionConfig:
    """Configuration for a v2 session."""

    def __init__(
        self,
        # Sampling
        k_samples: int = 5,  # Thoughts to sample per iteration
        # Pool parameters
        active_pool_size: int = 50,
        draft_buffer_size: int = 5,
        history_pairs: int = 10,
        # LLM
        model: str = "anthropic/claude-haiku-4.5",
        token_limit: int = 4000,
        # Embeddings
        embedding_model: str = "openai/text-embedding-3-small",
        embedding_dim: int = 1536,
        # Clustering
        min_cluster_size: int = 3,
        centroid_match_threshold: float = 0.3,
    ):
        self.k_samples = k_samples
        self.active_pool_size = active_pool_size
        self.draft_buffer_size = draft_buffer_size
        self.history_pairs = history_pairs
        self.model = model
        self.token_limit = token_limit
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.min_cluster_size = min_cluster_size
        self.centroid_match_threshold = centroid_match_threshold

    def to_dict(self) -> dict:
        """Convert to YAML-serializable dict."""
        return {
            'k_samples': self.k_samples,
            'active_pool_size': self.active_pool_size,
            'draft_buffer_size': self.draft_buffer_size,
            'history_pairs': self.history_pairs,
            'model': self.model,
            'token_limit': self.token_limit,
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,
            'min_cluster_size': self.min_cluster_size,
            'centroid_match_threshold': self.centroid_match_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionConfig:
        """Create from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__init__.__code__.co_varnames})


class SessionV2:
    """
    Manages a Logosphere v2 session with thinking pool and dialogue pool.

    Provides:
    - Unified access to thinking_pool and dialogue_pool
    - Iteration tracking
    - Configuration management
    - Intervention logging (audit trail)
    """

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)

        # Paths
        self._session_path = self.session_dir / 'session.yaml'
        self._thinking_dir = self.session_dir / 'thinking'
        self._dialogue_dir = self.session_dir / 'dialogue'

        # State
        self.iteration: int = 0
        self.config: SessionConfig = SessionConfig()

        # Pools (lazy loaded)
        self._thinking_pool: Optional[ThinkingPool] = None
        self._dialogue_pool: Optional[DialoguePool] = None

        # Intervention log
        self.intervention_log = InterventionLog(self.session_dir / 'interventions.jsonl')

        # Load or initialize
        if self._session_path.exists():
            self._load()
        else:
            self._init()

    @property
    def thinking_pool(self) -> ThinkingPool:
        """Get thinking pool (lazy loaded)."""
        if self._thinking_pool is None:
            self._thinking_pool = ThinkingPool(
                pool_dir=self._thinking_dir,
                active_pool_size=self.config.active_pool_size,
                embedding_dim=self.config.embedding_dim,
            )
        return self._thinking_pool

    @property
    def dialogue_pool(self) -> DialoguePool:
        """Get dialogue pool (lazy loaded)."""
        if self._dialogue_pool is None:
            self._dialogue_pool = DialoguePool(
                pool_dir=self._dialogue_dir,
                draft_buffer_size=self.config.draft_buffer_size,
                history_pairs=self.config.history_pairs,
            )
        return self._dialogue_pool

    def _init(self) -> None:
        """Initialize new session."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._thinking_dir.mkdir(exist_ok=True)
        self._dialogue_dir.mkdir(exist_ok=True)
        self._save()

    def _load(self) -> None:
        """Load existing session."""
        with open(self._session_path) as f:
            data = yaml.safe_load(f)

        self.iteration = data.get('iteration', 0)
        if 'config' in data:
            self.config = SessionConfig.from_dict(data['config'])

    def _save(self) -> None:
        """Save session state."""
        data = {
            'iteration': self.iteration,
            'config': self.config.to_dict(),
        }

        # Atomic write
        with tempfile.NamedTemporaryFile(
            mode='w', dir=self.session_dir, suffix='.tmp', delete=False
        ) as f:
            temp_path = Path(f.name)
            yaml.safe_dump(data, f, default_flow_style=False)

        shutil.move(str(temp_path), str(self._session_path))

    def save(self) -> None:
        """Save all session state (config + pools)."""
        self._save()
        if self._thinking_pool is not None:
            self._thinking_pool.save()
        if self._dialogue_pool is not None:
            self._dialogue_pool.save()

    # -------------------------------------------------------------------------
    # Thinking pool operations
    # -------------------------------------------------------------------------

    def add_thought(
        self,
        text: str,
        embedding,  # np.ndarray
        cluster: Optional[str] = None,
    ) -> int:
        """Add a thought to the thinking pool."""
        return self.thinking_pool.add(
            text=text,
            embedding=embedding,
            iter=self.iteration,
            cluster=cluster,
        )

    def sample_thoughts(self, k: int) -> tuple[list[Thought], list[int]]:
        """Sample k thoughts from thinking pool."""
        return self.thinking_pool.sample(k)

    # -------------------------------------------------------------------------
    # Dialogue pool operations
    # -------------------------------------------------------------------------

    def send_message(self, text: str) -> None:
        """
        User sends a message.

        Increments iteration (user's "thinking time") and enters drafting state.

        Raises:
            RuntimeError: If there are pending drafts (must accept first)
        """
        self.iteration += 1
        self.dialogue_pool.send_message(text=text, iter=self.iteration)

    def add_draft(self, text: str) -> None:
        """Mind produces a draft response."""
        self.dialogue_pool.add_draft(text=text, iter=self.iteration)

    def accept_draft(self, index: int = 1) -> Draft:
        """
        Accept a draft, moving the exchange to history.

        Args:
            index: 1-based index (1=latest, default)

        Returns:
            The accepted draft
        """
        return self.dialogue_pool.accept(index)

    def mark_drafts_seen(self, indices: Optional[list[int]] = None) -> None:
        """
        Mark drafts as seen.

        Args:
            indices: 1-based indices (1=latest). None means mark all.
        """
        self.dialogue_pool.mark_seen(indices)

    @property
    def is_drafting(self) -> bool:
        """True if there's a user message awaiting response."""
        return self.dialogue_pool.is_drafting

    def get_drafts(self) -> list[tuple[int, Draft]]:
        """Get drafts for display (newest first, 1-indexed)."""
        return self.dialogue_pool.get_drafts_for_display()

    def get_history(self) -> list[HistoryEntry]:
        """Get conversation history."""
        return self.dialogue_pool.get_history()

    # -------------------------------------------------------------------------
    # Clustering compatibility
    # -------------------------------------------------------------------------

    def get_visible_ids(self) -> set[int]:
        """Get all thought vector IDs (for clustering)."""
        return self.thinking_pool.get_visible_ids()

    # -------------------------------------------------------------------------
    # Class methods
    # -------------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        session_dir: Path,
        initial_prompt: Optional[str] = None,
        config: Optional[SessionConfig] = None,
    ) -> SessionV2:
        """
        Create a new session.

        Args:
            session_dir: Directory for the session
            initial_prompt: Optional seed message from user
            config: Optional custom configuration

        Returns:
            New SessionV2 instance
        """
        session = cls(session_dir)

        if config:
            session.config = config
            session._save()

        if initial_prompt:
            session.send_message(initial_prompt)
            session.save()

        return session

    @classmethod
    def exists(cls, session_dir: Path) -> bool:
        """Check if a session exists at the given path."""
        return (Path(session_dir) / 'session.yaml').exists()
