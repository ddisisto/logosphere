"""
DraftStore - Append-only draft storage for dialogue.

Stores all drafts with 0-indexed indices per message round.
Drafts are grouped by user_message_iter for easy querying.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


# Custom YAML dumper that uses literal block style for multiline strings
class _LiteralDumper(yaml.SafeDumper):
    pass


def _str_representer(dumper, data):
    """Use literal block style (|) for strings containing newlines."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralDumper.add_representer(str, _str_representer)


@dataclass
class Draft:
    """A draft response from the mind."""

    user_message_iter: int  # Which user message this responds to
    index: int  # 0-indexed within message round
    iter: int  # Global iteration when created
    time: str
    text: str
    seen: bool = False

    def to_dict(self) -> dict:
        return {
            "user_message_iter": self.user_message_iter,
            "index": self.index,
            "iter": self.iter,
            "time": self.time,
            "text": self.text,
            "seen": self.seen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Draft:
        return cls(
            user_message_iter=data["user_message_iter"],
            index=data["index"],
            iter=data["iter"],
            time=data["time"],
            text=data["text"],
            seen=data.get("seen", False),
        )


class DraftStore:
    """
    Append-only draft storage.

    Drafts are indexed by (user_message_iter, index) where:
    - user_message_iter: iteration of the user message being responded to
    - index: 0-based sequential index within that message round
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._drafts: list[Draft] = []

        if self._path.exists():
            self._load()

    def add_draft(
        self,
        user_message_iter: int,
        text: str,
        iter: int,
        seen: bool = False,
    ) -> Draft:
        """
        Add a draft for a user message.

        Index is automatically assigned as the next sequential index
        for this user_message_iter.

        Returns:
            The draft with assigned index
        """
        index = self.next_index(user_message_iter)

        draft = Draft(
            user_message_iter=user_message_iter,
            index=index,
            iter=iter,
            time=datetime.now(timezone.utc).isoformat(),
            text=text,
            seen=seen,
        )

        self._drafts.append(draft)
        return draft

    def mark_seen(
        self,
        user_message_iter: int,
        indices: list[int] | None = None,
    ) -> int:
        """
        Mark drafts as seen.

        Args:
            user_message_iter: Which message's drafts to mark
            indices: Specific indices to mark, or None for all

        Returns:
            Count of drafts marked
        """
        count = 0
        for draft in self._drafts:
            if draft.user_message_iter != user_message_iter:
                continue
            if draft.seen:
                continue
            if indices is None or draft.index in indices:
                draft.seen = True
                count += 1
        return count

    def get_drafts_for_message(self, user_message_iter: int) -> list[Draft]:
        """
        Get all drafts for a user message (oldest first).

        Args:
            user_message_iter: The user message iteration

        Returns:
            List of drafts sorted by index
        """
        drafts = [d for d in self._drafts if d.user_message_iter == user_message_iter]
        return sorted(drafts, key=lambda d: d.index)

    def get_draft(self, user_message_iter: int, index: int) -> Draft | None:
        """
        Get a specific draft by index.

        Args:
            user_message_iter: The user message iteration
            index: The draft index (0-based)

        Returns:
            The draft or None if not found
        """
        for draft in self._drafts:
            if draft.user_message_iter == user_message_iter and draft.index == index:
                return draft
        return None

    def get_latest_draft(self, user_message_iter: int) -> Draft | None:
        """
        Get the most recent draft for a message.

        Args:
            user_message_iter: The user message iteration

        Returns:
            The latest draft or None if no drafts
        """
        drafts = self.get_drafts_for_message(user_message_iter)
        return drafts[-1] if drafts else None

    def get_drafts_for_display(
        self,
        user_message_iter: int,
        max_count: int,
        max_chars: int,
    ) -> list[Draft]:
        """
        Get recent drafts within limits (newest first).

        Args:
            user_message_iter: The user message iteration
            max_count: Maximum number of drafts
            max_chars: Maximum total characters

        Returns:
            List of drafts (newest first) within limits
        """
        all_drafts = self.get_drafts_for_message(user_message_iter)

        result = []
        total_chars = 0

        # Iterate newest first
        for draft in reversed(all_drafts):
            if len(result) >= max_count:
                break
            if total_chars + len(draft.text) > max_chars and result:
                break
            result.append(draft)
            total_chars += len(draft.text)

        return result

    def next_index(self, user_message_iter: int) -> int:
        """
        Get the next draft index for a message.

        Args:
            user_message_iter: The user message iteration

        Returns:
            The next available index (0 if no drafts yet)
        """
        drafts = self.get_drafts_for_message(user_message_iter)
        if not drafts:
            return 0
        return max(d.index for d in drafts) + 1

    def save(self) -> None:
        """Save drafts to disk with atomic write."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        data = [d.to_dict() for d in self._drafts]

        with tempfile.NamedTemporaryFile(
            mode="w", dir=self._path.parent, suffix=".tmp", delete=False
        ) as f:
            temp_path = Path(f.name)
            yaml.dump(
                data,
                f,
                Dumper=_LiteralDumper,
                default_flow_style=False,
                allow_unicode=True,
            )

        shutil.move(str(temp_path), str(self._path))

    def _load(self) -> None:
        """Load drafts from disk."""
        with open(self._path) as f:
            data = yaml.safe_load(f) or []

        self._drafts = [Draft.from_dict(d) for d in data]
