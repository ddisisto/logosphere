"""
MessageStore - Append-only message storage for dialogue.

Stores user messages and accepted mind responses in a single file.
State is derived from the last message (no separate "awaiting" field).
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

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
class Message:
    """A message in the conversation."""

    role: Literal["user", "mind"]
    iter: int
    time: str
    text: str
    # Only for mind messages:
    draft_index: int | None = None  # Which draft was accepted (0-indexed)
    user_message_iter: int | None = None  # Links to the user message

    def to_dict(self) -> dict:
        d = {
            "role": self.role,
            "iter": self.iter,
            "time": self.time,
            "text": self.text,
        }
        if self.draft_index is not None:
            d["draft_index"] = self.draft_index
        if self.user_message_iter is not None:
            d["user_message_iter"] = self.user_message_iter
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        return cls(
            role=data["role"],
            iter=data["iter"],
            time=data["time"],
            text=data["text"],
            draft_index=data.get("draft_index"),
            user_message_iter=data.get("user_message_iter"),
        )


class MessageStore:
    """
    Append-only message storage.

    State derivation:
    - If last message is role: user -> DRAFTING state
    - If last message is role: mind or empty -> IDLE state
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._messages: list[Message] = []

        if self._path.exists():
            self._load()

    @property
    def is_drafting(self) -> bool:
        """True if last message is from user (awaiting response)."""
        if not self._messages:
            return False
        return self._messages[-1].role == "user"

    @property
    def last_user_message(self) -> Message | None:
        """Get the last user message, if in drafting state."""
        if not self.is_drafting:
            return None
        return self._messages[-1]

    @property
    def last_message(self) -> Message | None:
        """Get the last message of any role."""
        return self._messages[-1] if self._messages else None

    def add_user_message(self, text: str, iter: int) -> Message:
        """
        Add a user message.

        Raises:
            RuntimeError: If already in drafting state
        """
        if self.is_drafting:
            raise RuntimeError(
                "Cannot send message while awaiting response. "
                "Accept a draft first."
            )

        msg = Message(
            role="user",
            iter=iter,
            time=datetime.now(timezone.utc).isoformat(),
            text=text,
        )
        self._messages.append(msg)
        return msg

    def add_mind_message(
        self,
        text: str,
        iter: int,
        draft_index: int,
        user_message_iter: int,
    ) -> Message:
        """
        Add an accepted mind response.

        Raises:
            RuntimeError: If not in drafting state
        """
        if not self.is_drafting:
            raise RuntimeError("Cannot add mind message when not drafting.")

        msg = Message(
            role="mind",
            iter=iter,
            time=datetime.now(timezone.utc).isoformat(),
            text=text,
            draft_index=draft_index,
            user_message_iter=user_message_iter,
        )
        self._messages.append(msg)
        return msg

    def get_all(self) -> list[Message]:
        """Get all messages (oldest first)."""
        return list(self._messages)

    def get_history_for_display(
        self,
        max_count: int,
        max_chars: int,
    ) -> list[Message]:
        """
        Get recent history within limits.

        Excludes the current user message if in drafting state
        (that's shown separately as "awaiting").

        Args:
            max_count: Maximum number of messages
            max_chars: Maximum total characters

        Returns:
            List of messages (oldest first within returned subset)
        """
        # Get completed exchanges only (exclude pending user message)
        if self.is_drafting:
            candidates = self._messages[:-1]
        else:
            candidates = self._messages

        # Apply count limit (take from end)
        if len(candidates) > max_count:
            candidates = candidates[-max_count:]

        # Apply char limit (newest first, then reverse)
        result = []
        total_chars = 0

        for msg in reversed(candidates):
            if total_chars + len(msg.text) > max_chars and result:
                break
            result.append(msg)
            total_chars += len(msg.text)

        result.reverse()
        return result

    def save(self) -> None:
        """Save messages to disk with atomic write."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        data = [m.to_dict() for m in self._messages]

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
        """Load messages from disk."""
        with open(self._path) as f:
            data = yaml.safe_load(f) or []

        self._messages = [Message.from_dict(d) for d in data]
