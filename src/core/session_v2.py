"""
Session v2 - Dual-pool session management for Logosphere v2.

Manages thinking_pool and dialogue (messages + drafts) with YAML-based config.

Session structure:
    session/
    ├── session.yaml          # Config + iteration state
    ├── thinking/
    │   ├── embeddings.npy
    │   └── pool.yaml
    ├── messages.yaml         # User messages + accepted responses
    ├── drafts.yaml           # All drafts (0-indexed per message round)
    ├── clusters/             # Unchanged from v1
    ├── prompts/              # LLM request/response logs (created by runner)
    │   ├── {iter:06d}-req.txt
    │   └── {iter:06d}-resp.txt
    └── interventions.jsonl   # Audit trail
"""

from __future__ import annotations

import tempfile
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal

import yaml

from .thinking_pool import ThinkingPool, Thought
from .message_store import MessageStore, Message
from .draft_store import DraftStore, Draft
from .intervention_log import InterventionLog


# -----------------------------------------------------------------------------
# User Signal
# -----------------------------------------------------------------------------

PresenceState = Literal['absent', 'reviewing', 'engaged']


@dataclass
class UserSignal:
    """A user presence/status signal entry."""
    iter: int
    presence: PresenceState
    status: str
    time: str  # "Day HH:MM" format

    def to_dict(self) -> dict:
        return {
            'iter': self.iter,
            'presence': self.presence,
            'status': self.status,
            'time': self.time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UserSignal:
        return cls(
            iter=data['iter'],
            presence=data['presence'],
            status=data.get('status', ''),
            time=data.get('time', ''),
        )

    @staticmethod
    def format_local_time() -> str:
        """Format current local time as 'Day HH:MM'."""
        now = datetime.now()
        return now.strftime('%a %H:%M')


class SessionConfig:
    """Configuration for a v2 session."""

    def __init__(
        self,
        # Thought display limits (sampled from FIFO pool)
        thought_display_chars: int = 3000,  # Max chars of sampled thoughts
        thought_display_count: int = 10,  # Max thoughts to sample per iteration
        # Pool parameters
        active_pool_size: int = 50,
        # Draft display limits (storage is unlimited)
        draft_display_chars: int = 2000,  # Show drafts up to this many chars
        draft_display_count: int = 16,  # Show at most this many drafts
        # History display limits (storage is unlimited)
        history_display_chars: int = 4000,  # Max chars of history to show mind
        history_display_count: int = 20,  # Max history entries to show mind
        # LLM
        model: str = "anthropic/claude-haiku-4.5",
        token_limit: int = 4000,
        # Embeddings
        embedding_model: str = "openai/text-embedding-3-small",
        embedding_dim: int = 1536,
        # Clustering
        min_cluster_size: int = 3,
        centroid_match_threshold: float = 0.3,
        # Signal handling
        hard_signal_threshold: int = 3,  # Consecutive hard signals before forced stop
    ):
        self.thought_display_chars = thought_display_chars
        self.thought_display_count = thought_display_count
        self.active_pool_size = active_pool_size
        self.draft_display_chars = draft_display_chars
        self.draft_display_count = draft_display_count
        self.history_display_chars = history_display_chars
        self.history_display_count = history_display_count
        self.model = model
        self.token_limit = token_limit
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.min_cluster_size = min_cluster_size
        self.centroid_match_threshold = centroid_match_threshold
        self.hard_signal_threshold = hard_signal_threshold

    def to_dict(self) -> dict:
        """Convert to YAML-serializable dict."""
        return {
            'thought_display_chars': self.thought_display_chars,
            'thought_display_count': self.thought_display_count,
            'active_pool_size': self.active_pool_size,
            'draft_display_chars': self.draft_display_chars,
            'draft_display_count': self.draft_display_count,
            'history_display_chars': self.history_display_chars,
            'history_display_count': self.history_display_count,
            'model': self.model,
            'token_limit': self.token_limit,
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,
            'min_cluster_size': self.min_cluster_size,
            'centroid_match_threshold': self.centroid_match_threshold,
            'hard_signal_threshold': self.hard_signal_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionConfig:
        """Create from dict with migration support for old param names."""
        # Create a copy to avoid modifying the input
        migrated = dict(data)

        # Migration: k_samples -> thought_display_count
        if 'k_samples' in migrated and 'thought_display_count' not in migrated:
            migrated['thought_display_count'] = migrated.pop('k_samples')
        elif 'k_samples' in migrated:
            del migrated['k_samples']

        # Migration: history_display_pairs -> history_display_count (multiply by 2)
        if 'history_display_pairs' in migrated and 'history_display_count' not in migrated:
            migrated['history_display_count'] = migrated.pop('history_display_pairs') * 2
        elif 'history_display_pairs' in migrated:
            del migrated['history_display_pairs']

        # Filter to only valid params
        valid_params = cls.__init__.__code__.co_varnames
        return cls(**{k: v for k, v in migrated.items() if k in valid_params})


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
        self._messages_path = self.session_dir / 'messages.yaml'
        self._drafts_path = self.session_dir / 'drafts.yaml'

        # State
        self.iteration: int = 0
        self.config: SessionConfig = SessionConfig()
        self.user_signal: list[UserSignal] = []  # Append-only history

        # Pools/stores (lazy loaded)
        self._thinking_pool: Optional[ThinkingPool] = None
        self._messages: Optional[MessageStore] = None
        self._drafts: Optional[DraftStore] = None

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
    def messages(self) -> MessageStore:
        """Get message store (lazy loaded)."""
        if self._messages is None:
            self._messages = MessageStore(self._messages_path)
        return self._messages

    @property
    def drafts(self) -> DraftStore:
        """Get draft store (lazy loaded)."""
        if self._drafts is None:
            self._drafts = DraftStore(self._drafts_path)
        return self._drafts

    def _init(self) -> None:
        """Initialize new session."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._thinking_dir.mkdir(exist_ok=True)
        # Default user signal: absent with empty status
        self.user_signal = [UserSignal(
            iter=0,
            presence='absent',
            status='',
            time=UserSignal.format_local_time(),
        )]
        self._save()

    def _load(self) -> None:
        """Load existing session."""
        with open(self._session_path) as f:
            data = yaml.safe_load(f)

        self.iteration = data.get('iteration', 0)
        if 'config' in data:
            self.config = SessionConfig.from_dict(data['config'])
        # Load user signal history
        if 'user_signal' in data:
            self.user_signal = [UserSignal.from_dict(s) for s in data['user_signal']]
        else:
            # Migration: old session without signal, default to absent
            self.user_signal = [UserSignal(
                iter=0,
                presence='absent',
                status='',
                time='',
            )]

    def _save(self) -> None:
        """Save session state."""
        data = {
            'iteration': self.iteration,
            'config': self.config.to_dict(),
            'user_signal': [s.to_dict() for s in self.user_signal],
        }

        # Atomic write
        with tempfile.NamedTemporaryFile(
            mode='w', dir=self.session_dir, suffix='.tmp', delete=False
        ) as f:
            temp_path = Path(f.name)
            yaml.safe_dump(data, f, default_flow_style=False)

        shutil.move(str(temp_path), str(self._session_path))

    def save(self) -> None:
        """Save all session state (config + stores)."""
        self._save()
        if self._thinking_pool is not None:
            self._thinking_pool.save()
        if self._messages is not None:
            self._messages.save()
        if self._drafts is not None:
            self._drafts.save()

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

    def sample_thoughts(self) -> tuple[list[Thought], list[int]]:
        """Sample thoughts from thinking pool within display limits."""
        return self.thinking_pool.sample(
            k=self.config.thought_display_count,
            max_chars=self.config.thought_display_chars,
        )

    # -------------------------------------------------------------------------
    # Dialogue operations (messages + drafts)
    # -------------------------------------------------------------------------

    def send_message(self, text: str) -> Message:
        """
        User sends a message.

        Increments iteration (user's "thinking time") and enters drafting state.

        Raises:
            RuntimeError: If already in drafting state (must accept first)

        Returns:
            The created message
        """
        self.iteration += 1
        return self.messages.add_user_message(text=text, iter=self.iteration)

    def add_draft(self, text: str, seen: bool = False) -> Draft:
        """
        Mind produces a draft response.

        Args:
            text: Draft content
            seen: Mark as seen immediately

        Returns:
            The created draft with assigned index
        """
        user_msg = self.messages.last_user_message
        if user_msg is None:
            raise RuntimeError("Cannot add draft when not drafting")

        return self.drafts.add_draft(
            user_message_iter=user_msg.iter,
            text=text,
            iter=self.iteration,
            seen=seen,
        )

    def accept_draft(self, index: int) -> Draft:
        """
        Accept a draft by index, completing the exchange.

        Args:
            index: 0-based draft index

        Returns:
            The accepted draft

        Raises:
            RuntimeError: If not in drafting state
            IndexError: If draft not found
        """
        user_msg = self.messages.last_user_message
        if user_msg is None:
            raise RuntimeError("Cannot accept draft when not drafting")

        draft = self.drafts.get_draft(user_msg.iter, index)
        if draft is None:
            valid = [d.index for d in self.get_current_drafts()]
            raise IndexError(f"Draft {index} not found. Valid: {valid}")

        self.messages.add_mind_message(
            text=draft.text,
            iter=draft.iter,
            draft_index=index,
            user_message_iter=user_msg.iter,
        )
        return draft

    def mark_drafts_seen(self, indices: Optional[list[int]] = None) -> int:
        """
        Mark drafts as seen.

        Args:
            indices: 0-based indices. None means mark all.

        Returns:
            Count of drafts marked
        """
        user_msg = self.messages.last_user_message
        if user_msg is None:
            return 0

        return self.drafts.mark_seen(user_msg.iter, indices)

    @property
    def is_drafting(self) -> bool:
        """True if there's a user message awaiting response."""
        return self.messages.is_drafting

    def get_current_drafts(self) -> list[Draft]:
        """Get all drafts for current message round (oldest first)."""
        user_msg = self.messages.last_user_message
        if user_msg is None:
            return []
        return self.drafts.get_drafts_for_message(user_msg.iter)

    def get_drafts_for_mind(self) -> list[Draft]:
        """Get drafts for display to mind (newest first, within limits)."""
        user_msg = self.messages.last_user_message
        if user_msg is None:
            return []
        return self.drafts.get_drafts_for_display(
            user_message_iter=user_msg.iter,
            max_count=self.config.draft_display_count,
            max_chars=self.config.draft_display_chars,
        )

    def get_all_drafts(self) -> list[Draft]:
        """Get all drafts for current message (oldest first) for user display."""
        return self.get_current_drafts()

    def get_history(self) -> list[Message]:
        """Get all conversation history (for CLI/analysis)."""
        return self.messages.get_all()

    def get_history_for_mind(self) -> list[Message]:
        """Get display-limited history for mind input."""
        return self.messages.get_history_for_display(
            max_count=self.config.history_display_count,
            max_chars=self.config.history_display_chars,
        )

    def get_awaiting_message(self) -> Message | None:
        """Get the current user message awaiting response, if any."""
        return self.messages.last_user_message

    # -------------------------------------------------------------------------
    # User Signal
    # -------------------------------------------------------------------------

    def add_user_signal(
        self,
        presence: Optional[PresenceState] = None,
        status: Optional[str] = None,
    ) -> UserSignal:
        """
        Add a user signal entry (presence and/or status update).

        Args:
            presence: New presence state (None = keep current)
            status: New status text (None = keep current)

        Returns:
            The new signal entry
        """
        latest = self.get_latest_signal()

        signal = UserSignal(
            iter=self.iteration,
            presence=presence if presence is not None else latest.presence,
            status=status if status is not None else latest.status,
            time=UserSignal.format_local_time(),
        )
        self.user_signal.append(signal)
        return signal

    def get_latest_signal(self) -> UserSignal:
        """Get the most recent user signal."""
        if self.user_signal:
            return self.user_signal[-1]
        # Fallback (shouldn't happen after init)
        return UserSignal(iter=0, presence='absent', status='', time='')

    def get_signals_for_mind(self, max_count: int = 3) -> list[UserSignal]:
        """
        Get recent signals for display to mind.

        Args:
            max_count: Maximum signals to return (default 3)

        Returns:
            List of signals, newest first
        """
        return list(reversed(self.user_signal[-max_count:]))

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
