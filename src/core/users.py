"""
User management for Logosphere v2.

Manages multiple users with individual presence/status state.

Storage: {session}/users.yaml
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal

import yaml


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

PresenceState = Literal['absent', 'reviewing', 'engaged']

PRESENCE_CYCLE = ['absent', 'reviewing', 'engaged']


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class UserStateEntry:
    """A single state entry for a user."""
    iter: int
    presence: PresenceState
    status: str = ""
    time: str = ""  # "Day HH:MM" format, empty if show_clock=False

    def to_dict(self) -> dict:
        d = {
            'iter': self.iter,
            'presence': self.presence,
        }
        if self.status:
            d['status'] = self.status
        if self.time:
            d['time'] = self.time
        return d

    @classmethod
    def from_dict(cls, data: dict) -> UserStateEntry:
        return cls(
            iter=data['iter'],
            presence=data['presence'],
            status=data.get('status', ''),
            time=data.get('time', ''),
        )

    def to_flow_str(self) -> str:
        """Format as YAML flow style for compact display."""
        parts = [f"age: {{age}}"]  # Placeholder, filled by caller
        parts.append(f"presence: {self.presence}")
        if self.status:
            parts.append(f'status: "{self.status}"')
        if self.time:
            parts.append(f'time: "{self.time}"')
        return "{" + ", ".join(parts) + "}"


@dataclass
class User:
    """A user with identity and state history."""
    id: str
    name: str
    show_clock: bool = True
    state_display_count: int = 3
    state: list[UserStateEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'show_clock': self.show_clock,
            'state_display_count': self.state_display_count,
            'state': [s.to_dict() for s in self.state],
        }

    @classmethod
    def from_dict(cls, data: dict) -> User:
        return cls(
            id=data['id'],
            name=data['name'],
            show_clock=data.get('show_clock', True),
            state_display_count=data.get('state_display_count', 3),
            state=[UserStateEntry.from_dict(s) for s in data.get('state', [])],
        )

    def get_latest_state(self) -> Optional[UserStateEntry]:
        """Get most recent state entry."""
        return self.state[-1] if self.state else None

    def get_state_for_display(self, current_iter: int) -> list[tuple[int, UserStateEntry]]:
        """
        Get state entries for display to mind.

        Returns:
            List of (age, entry) tuples, newest first, up to state_display_count
        """
        entries = []
        for entry in reversed(self.state[-self.state_display_count:]):
            age = current_iter - entry.iter
            entries.append((age, entry))
        return entries

    def add_state(
        self,
        iter: int,
        presence: Optional[PresenceState] = None,
        status: Optional[str] = None,
        show_clock: bool = True,
    ) -> UserStateEntry:
        """
        Add a new state entry.

        Args:
            iter: Current iteration
            presence: New presence (None = keep current)
            status: New status (None = keep current)
            show_clock: Whether to include time

        Returns:
            The new state entry
        """
        latest = self.get_latest_state()

        entry = UserStateEntry(
            iter=iter,
            presence=presence if presence is not None else (latest.presence if latest else 'absent'),
            status=status if status is not None else (latest.status if latest else ''),
            time=format_local_time() if show_clock and self.show_clock else '',
        )
        self.state.append(entry)
        return entry

    def cycle_presence(self, iter: int) -> UserStateEntry:
        """Cycle to next presence state."""
        latest = self.get_latest_state()
        current = latest.presence if latest else 'absent'
        idx = PRESENCE_CYCLE.index(current)
        next_presence = PRESENCE_CYCLE[(idx + 1) % len(PRESENCE_CYCLE)]
        return self.add_state(iter=iter, presence=next_presence)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

class UserRegistry:
    """
    Manages users for a session.

    Storage: {session_dir}/users.yaml
    """

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self._path = self.session_dir / 'users.yaml'
        self._users: dict[str, User] = {}
        self._last_user_id: str = ""
        self._load()

    def _load(self) -> None:
        """Load users from disk."""
        if self._path.exists():
            with open(self._path) as f:
                data = yaml.safe_load(f) or {}
            self._last_user_id = data.get('last_user_id', '')
            for user_data in data.get('users', []):
                user = User.from_dict(user_data)
                self._users[user.id] = user

    def save(self) -> None:
        """Save users to disk."""
        data = {
            'last_user_id': self._last_user_id,
            'users': [u.to_dict() for u in self._users.values()],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    @property
    def last_user_id(self) -> str:
        """Get last active user ID."""
        return self._last_user_id

    @last_user_id.setter
    def last_user_id(self, value: str) -> None:
        """Set last active user ID."""
        self._last_user_id = value

    def get(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)

    def get_or_create(self, user_id: str, name: str = "") -> User:
        """Get existing user or create new one."""
        if user_id not in self._users:
            self._users[user_id] = User(
                id=user_id,
                name=name or user_id,
            )
        return self._users[user_id]

    def create(self, user_id: str, name: str = "") -> User:
        """
        Create a new user.

        Raises:
            ValueError: If user already exists
        """
        if user_id in self._users:
            raise ValueError(f"User '{user_id}' already exists")
        user = User(id=user_id, name=name or user_id)
        self._users[user_id] = user
        return user

    def list_users(self) -> list[User]:
        """Get all users."""
        return list(self._users.values())

    def user_ids(self) -> list[str]:
        """Get all user IDs."""
        return list(self._users.keys())

    def __len__(self) -> int:
        return len(self._users)

    def __bool__(self) -> bool:
        return bool(self._users)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def format_local_time() -> str:
    """Format current local time as 'Day HH:MM'."""
    now = datetime.now()
    return now.strftime('%a %H:%M')


def parse_presence_shorthand(value: str) -> Optional[PresenceState]:
    """
    Parse presence value with shorthand support.

    Args:
        value: 'absent', 'reviewing', 'engaged' or 'a', 'r', 'e'

    Returns:
        PresenceState or None if invalid
    """
    shortcuts = {'a': 'absent', 'r': 'reviewing', 'e': 'engaged'}
    parsed = shortcuts.get(value.lower(), value.lower())
    if parsed in ('absent', 'reviewing', 'engaged'):
        return parsed  # type: ignore
    return None
