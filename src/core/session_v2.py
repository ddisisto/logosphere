"""
Session v2 - Dual-pool session management for Logosphere v2.

Manages both thinking_pool and message_pool, with YAML-based config.

Session structure:
    session/
    ├── session.yaml          # Config + iteration state
    ├── thinking/
    │   ├── embeddings.npy
    │   └── pool.yaml
    ├── messages/
    │   └── pool.yaml
    ├── clusters/             # Unchanged from v1
    └── interventions.jsonl   # Audit trail (kept from v1)
"""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path
from typing import Optional

import yaml

from .thinking_pool import ThinkingPool, Thought
from .message_pool import MessagePool, Message
from .intervention_log import InterventionLog


class SessionConfig:
    """Configuration for a v2 session."""

    def __init__(
        self,
        # Pool parameters
        active_pool_size: int = 50,
        message_buffer_per_source: int = 10,
        # LLM
        model: str = "anthropic/claude-haiku-4.5",
        token_limit: int = 4000,
        # Embeddings
        embedding_model: str = "openai/text-embedding-3-small",
        embedding_dim: int = 1536,
        # Clustering
        min_cluster_size: int = 3,
        centroid_match_threshold: float = 0.3,
        noise_reconsider_iterations: int = 20,
    ):
        self.active_pool_size = active_pool_size
        self.message_buffer_per_source = message_buffer_per_source
        self.model = model
        self.token_limit = token_limit
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.min_cluster_size = min_cluster_size
        self.centroid_match_threshold = centroid_match_threshold
        self.noise_reconsider_iterations = noise_reconsider_iterations

    def to_dict(self) -> dict:
        """Convert to YAML-serializable dict."""
        return {
            'active_pool_size': self.active_pool_size,
            'message_buffer_per_source': self.message_buffer_per_source,
            'model': self.model,
            'token_limit': self.token_limit,
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,
            'min_cluster_size': self.min_cluster_size,
            'centroid_match_threshold': self.centroid_match_threshold,
            'noise_reconsider_iterations': self.noise_reconsider_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionConfig:
        """Create from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__init__.__code__.co_varnames})


class SessionV2:
    """
    Manages a Logosphere v2 session with dual pools.

    Provides:
    - Unified access to thinking_pool and message_pool
    - Iteration tracking
    - Configuration management
    - Intervention logging (audit trail)
    """

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)

        # Paths
        self._session_path = self.session_dir / 'session.yaml'
        self._thinking_dir = self.session_dir / 'thinking'
        self._messages_dir = self.session_dir / 'messages'

        # State
        self.iteration: int = 0
        self.config: SessionConfig = SessionConfig()

        # Pools (lazy loaded)
        self._thinking_pool: Optional[ThinkingPool] = None
        self._message_pool: Optional[MessagePool] = None

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
    def message_pool(self) -> MessagePool:
        """Get message pool (lazy loaded)."""
        if self._message_pool is None:
            self._message_pool = MessagePool(
                pool_dir=self._messages_dir,
                buffer_size_per_source=self.config.message_buffer_per_source,
            )
        return self._message_pool

    def _init(self) -> None:
        """Initialize new session."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._thinking_dir.mkdir(exist_ok=True)
        self._messages_dir.mkdir(exist_ok=True)
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
        if self._message_pool is not None:
            self._message_pool.save()

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
    # Message pool operations
    # -------------------------------------------------------------------------

    def add_message(
        self,
        source: str,
        to: str,
        text: str,
    ) -> None:
        """Add a message to the message pool."""
        self.message_pool.add(
            source=source,
            to=to,
            iter=self.iteration,
            text=text,
        )

    def get_messages(self) -> list[Message]:
        """Get all active messages."""
        return self.message_pool.get_active()

    def get_messages_for(self, recipient: str) -> list[Message]:
        """Get messages addressed to a specific recipient."""
        return self.message_pool.get_active_for_recipient(recipient)

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
            session.add_message(
                source='user',
                to='mind_0',
                text=initial_prompt,
            )
            session.message_pool.save()

        return session

    @classmethod
    def exists(cls, session_dir: Path) -> bool:
        """Check if a session exists at the given path."""
        return (Path(session_dir) / 'session.yaml').exists()
