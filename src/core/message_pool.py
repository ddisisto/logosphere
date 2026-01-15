"""
MessagePool - Direct communication channel for Logosphere v2.

Stores messages between user and minds. No embeddings, no clustering.
Per-source FIFO rotation (each source has its own rolling buffer).

Uses YAML for human-readable storage.
"""

from __future__ import annotations

import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


class Message:
    """A single message in the message pool."""

    __slots__ = ('source', 'to', 'iter', 'time', 'text')

    def __init__(
        self,
        source: str,
        to: str,
        iter: int,
        time: str,
        text: str,
    ):
        self.source = source  # e.g., "user", "mind_0", "mind_1"
        self.to = to  # e.g., "user", "mind_0", or None for broadcast
        self.iter = iter
        self.time = time
        self.text = text

    def to_dict(self) -> dict:
        """Convert to YAML-serializable dict."""
        return {
            'source': self.source,
            'to': self.to,
            'iter': self.iter,
            'time': self.time,
            'text': self.text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        """Create from dict."""
        return cls(
            source=data['source'],
            to=data['to'],
            iter=data['iter'],
            time=data['time'],
            text=data['text'],
        )


class MessagePool:
    """
    Direct communication channel with per-source FIFO buffers.

    Storage format:
        messages/
        └── pool.yaml         # List of all messages (chronological)

    Each source has a configurable buffer size. When a source's
    buffer is full, oldest messages from that source are archived.

    Active pool = most recent N messages per source.
    """

    def __init__(
        self,
        pool_dir: Path,
        buffer_size_per_source: int = 10,
    ):
        self.pool_dir = Path(pool_dir)
        self.buffer_size_per_source = buffer_size_per_source

        # All messages (chronological order)
        self.messages: list[Message] = []

        # Archive (messages displaced from active buffers)
        self.archive: list[Message] = []

        # Paths
        self._pool_path = self.pool_dir / 'pool.yaml'
        self._archive_path = self.pool_dir / 'archive.yaml'

        # Load if exists
        if self._pool_path.exists():
            self._load()

    def add(
        self,
        source: str,
        to: str,
        iter: int,
        text: str,
    ) -> None:
        """
        Add a message to the pool.

        If source's buffer is full, oldest message from that source
        is moved to archive.

        Args:
            source: Who sent this (e.g., "user", "mind_0")
            to: Who it's addressed to
            iter: Current iteration
            text: Message content
        """
        message = Message(
            source=source,
            to=to,
            iter=iter,
            time=datetime.now(timezone.utc).isoformat(),
            text=text,
        )

        # Check if we need to archive old messages from this source
        source_messages = [m for m in self.messages if m.source == source]
        if len(source_messages) >= self.buffer_size_per_source:
            # Move oldest from this source to archive
            oldest = source_messages[0]
            self.messages.remove(oldest)
            self.archive.append(oldest)

        self.messages.append(message)

    def get_active(self) -> list[Message]:
        """
        Get all active messages (chronological order).

        Returns messages from all sources, respecting per-source limits.
        """
        return list(self.messages)

    def get_active_for_source(self, source: str) -> list[Message]:
        """Get active messages from a specific source."""
        return [m for m in self.messages if m.source == source]

    def get_active_for_recipient(self, to: str) -> list[Message]:
        """Get active messages addressed to a specific recipient."""
        return [m for m in self.messages if m.to == to]

    def get_archive(self) -> list[Message]:
        """Get all archived messages."""
        return list(self.archive)

    def size(self) -> int:
        """Total active messages."""
        return len(self.messages)

    def archive_size(self) -> int:
        """Total archived messages."""
        return len(self.archive)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self) -> None:
        """Save pool to disk with atomic writes."""
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        # Save active messages
        pool_data = [m.to_dict() for m in self.messages]
        self._atomic_yaml_write(self._pool_path, pool_data)

        # Save archive if non-empty
        if self.archive:
            archive_data = [m.to_dict() for m in self.archive]
            self._atomic_yaml_write(self._archive_path, archive_data)

    def _atomic_yaml_write(self, path: Path, data: list) -> None:
        """Write YAML atomically using temp file + rename."""
        with tempfile.NamedTemporaryFile(
            mode='w', dir=self.pool_dir, suffix='.tmp', delete=False
        ) as f:
            temp_path = Path(f.name)
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

        shutil.move(str(temp_path), str(path))

    def _load(self) -> None:
        """Load pool from disk."""
        # Load active messages
        with open(self._pool_path) as f:
            pool_data = yaml.safe_load(f) or []

        self.messages = [Message.from_dict(d) for d in pool_data]

        # Load archive if exists
        if self._archive_path.exists():
            with open(self._archive_path) as f:
                archive_data = yaml.safe_load(f) or []
            self.archive = [Message.from_dict(d) for d in archive_data]
