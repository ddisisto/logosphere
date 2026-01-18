"""
DialoguePool - Draft-based dialogue system for Logosphere v2.

Replaces MessagePool with a structured dialogue flow:
- User sends message → mind drafts responses → user accepts one
- Strict one-to-one: can't send new message until accepting a draft

Storage:
    dialogue/
    └── pool.yaml    # awaiting, drafts, history
"""

from __future__ import annotations

import tempfile
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


# Custom YAML dumper that uses literal block style for multiline strings
class _LiteralDumper(yaml.SafeDumper):
    pass


def _str_representer(dumper, data):
    """Use literal block style (|) for strings containing newlines."""
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


_LiteralDumper.add_representer(str, _str_representer)


@dataclass
class UserMessage:
    """A user message awaiting response."""
    iter: int
    time: str
    text: str

    def to_dict(self) -> dict:
        return {'iter': self.iter, 'time': self.time, 'text': self.text}

    @classmethod
    def from_dict(cls, data: dict) -> UserMessage:
        return cls(iter=data['iter'], time=data['time'], text=data['text'])


@dataclass
class Draft:
    """A draft response from the mind."""
    iter: int
    time: str
    text: str
    seen: bool = False

    def to_dict(self) -> dict:
        return {'iter': self.iter, 'time': self.time, 'text': self.text, 'seen': self.seen}

    @classmethod
    def from_dict(cls, data: dict) -> Draft:
        return cls(
            iter=data['iter'],
            time=data['time'],
            text=data['text'],
            seen=data.get('seen', False),
        )


@dataclass
class HistoryEntry:
    """An entry in conversation history (user message or accepted response)."""
    role: str  # "user" or "mind"
    iter: int
    time: str
    text: str

    def to_dict(self) -> dict:
        return {'role': self.role, 'iter': self.iter, 'time': self.time, 'text': self.text}

    @classmethod
    def from_dict(cls, data: dict) -> HistoryEntry:
        return cls(
            role=data['role'],
            iter=data['iter'],
            time=data['time'],
            text=data['text'],
        )


class DialoguePool:
    """
    Draft-based dialogue system.

    State machine:
    - IDLE: No awaiting message. History available for context.
    - DRAFTING: User message awaiting response. Drafts accumulate.

    Flow:
    1. user sends message → state becomes DRAFTING
    2. mind iterations produce drafts
    3. user marks drafts as seen (optional)
    4. user accepts a draft → state becomes IDLE, exchange added to history
    """

    def __init__(
        self,
        pool_dir: Path,
        draft_buffer_size: int = 5,
        history_pairs: int = 10,
    ):
        self.pool_dir = Path(pool_dir)
        self.draft_buffer_size = draft_buffer_size
        self.history_pairs = history_pairs  # Max user+response pairs to retain

        # State
        self.awaiting: Optional[UserMessage] = None
        self.drafts: list[Draft] = []  # Oldest first in storage
        self.history: list[HistoryEntry] = []  # Oldest first

        # Path
        self._pool_path = self.pool_dir / 'pool.yaml'

        # Load if exists
        if self._pool_path.exists():
            self._load()

    @property
    def is_drafting(self) -> bool:
        """True if there's a user message awaiting response."""
        return self.awaiting is not None

    def send_message(self, text: str, iter: int) -> None:
        """
        User sends a message.

        Raises:
            RuntimeError: If there are pending drafts (must accept first)
        """
        if self.is_drafting:
            raise RuntimeError(
                "Cannot send message while drafts are pending. "
                "Use 'mind accept' to accept a draft first."
            )

        self.awaiting = UserMessage(
            iter=iter,
            time=datetime.now(timezone.utc).isoformat(),
            text=text,
        )
        self.drafts = []

    def add_draft(self, text: str, iter: int) -> None:
        """
        Mind produces a draft response.

        If buffer is full, oldest draft is discarded.
        """
        if not self.is_drafting:
            # No awaiting message - this shouldn't happen in normal flow
            # but we'll allow it (draft goes nowhere useful)
            return

        draft = Draft(
            iter=iter,
            time=datetime.now(timezone.utc).isoformat(),
            text=text,
            seen=False,
        )

        self.drafts.append(draft)

        # Enforce buffer limit (drop oldest)
        if len(self.drafts) > self.draft_buffer_size:
            self.drafts = self.drafts[-self.draft_buffer_size:]

    def mark_seen(self, indices: Optional[list[int]] = None) -> None:
        """
        Mark drafts as seen.

        Args:
            indices: 1-based indices (1=latest). None means mark all.
        """
        if indices is None:
            for draft in self.drafts:
                draft.seen = True
        else:
            # Convert 1-based (newest first) to 0-based (oldest first in storage)
            for i in indices:
                if 1 <= i <= len(self.drafts):
                    # i=1 is latest = last in storage
                    storage_idx = len(self.drafts) - i
                    self.drafts[storage_idx].seen = True

    def accept(self, index: int = 1) -> Draft:
        """
        Accept a draft, moving the exchange to history.

        Args:
            index: 1-based index (1=latest, default)

        Returns:
            The accepted draft

        Raises:
            RuntimeError: If not in drafting state or no drafts
            IndexError: If index out of range
        """
        if not self.is_drafting:
            raise RuntimeError("No pending message to accept draft for.")

        if not self.drafts:
            raise RuntimeError("No drafts available to accept.")

        if not (1 <= index <= len(self.drafts)):
            raise IndexError(f"Draft index {index} out of range (1-{len(self.drafts)})")

        # Get the draft (1=latest = last in storage)
        storage_idx = len(self.drafts) - index
        accepted = self.drafts[storage_idx]

        # Add to history: user message then accepted response
        self.history.append(HistoryEntry(
            role='user',
            iter=self.awaiting.iter,
            time=self.awaiting.time,
            text=self.awaiting.text,
        ))
        self.history.append(HistoryEntry(
            role='mind',
            iter=accepted.iter,
            time=accepted.time,
            text=accepted.text,
        ))

        # Trim history (keep last N pairs = 2N entries)
        max_entries = self.history_pairs * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

        # Clear drafting state
        self.awaiting = None
        self.drafts = []

        return accepted

    def get_drafts_for_display(self) -> list[tuple[int, Draft]]:
        """
        Get drafts for display (newest first, 1-indexed).

        Returns:
            List of (display_index, draft) tuples
        """
        result = []
        for i, draft in enumerate(reversed(self.drafts)):
            result.append((i + 1, draft))
        return result

    def get_history(self) -> list[HistoryEntry]:
        """Get conversation history (oldest first)."""
        return list(self.history)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self) -> None:
        """Save pool to disk with atomic write."""
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        data = {
            'awaiting': self.awaiting.to_dict() if self.awaiting else None,
            'drafts': [d.to_dict() for d in self.drafts],
            'history': [h.to_dict() for h in self.history],
        }

        with tempfile.NamedTemporaryFile(
            mode='w', dir=self.pool_dir, suffix='.tmp', delete=False
        ) as f:
            temp_path = Path(f.name)
            yaml.dump(data, f, Dumper=_LiteralDumper,
                      default_flow_style=False, allow_unicode=True)

        shutil.move(str(temp_path), str(self._pool_path))

    def _load(self) -> None:
        """Load pool from disk."""
        with open(self._pool_path) as f:
            data = yaml.safe_load(f) or {}

        if data.get('awaiting'):
            self.awaiting = UserMessage.from_dict(data['awaiting'])

        self.drafts = [Draft.from_dict(d) for d in data.get('drafts', [])]
        self.history = [HistoryEntry.from_dict(h) for h in data.get('history', [])]
