"""
Session locking for Logosphere.

Prevents concurrent access from TUI and CLI to avoid race conditions.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


LOCK_FILENAME = ".lock"


@dataclass
class LockInfo:
    """Information about a session lock."""
    pid: int
    holder: str  # "tui", "cli:run", "cli:step", etc.
    started: str  # ISO format timestamp

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "holder": self.holder,
            "started": self.started,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LockInfo":
        return cls(
            pid=data["pid"],
            holder=data["holder"],
            started=data["started"],
        )


class LockError(Exception):
    """Raised when a lock cannot be acquired."""
    def __init__(self, message: str, lock_info: Optional[LockInfo] = None):
        super().__init__(message)
        self.lock_info = lock_info


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks
        return True
    except OSError:
        return False


def get_lock_path(session_dir: Path) -> Path:
    """Get the lock file path for a session."""
    return session_dir / LOCK_FILENAME


def read_lock(session_dir: Path) -> Optional[LockInfo]:
    """
    Read the current lock info, if any.

    Returns None if no lock exists or lock file is invalid.
    """
    lock_path = get_lock_path(session_dir)
    if not lock_path.exists():
        return None

    try:
        with open(lock_path) as f:
            data = yaml.safe_load(f)
        if data is None:
            return None
        return LockInfo.from_dict(data)
    except (yaml.YAMLError, KeyError, TypeError):
        return None


def is_lock_stale(lock_info: LockInfo) -> bool:
    """Check if a lock is stale (holder process no longer running)."""
    return not _is_pid_alive(lock_info.pid)


def acquire_lock(session_dir: Path, holder: str) -> LockInfo:
    """
    Acquire a lock on the session.

    Args:
        session_dir: Path to the session directory
        holder: Identifier for lock holder (e.g., "tui", "cli:run")

    Returns:
        LockInfo for the acquired lock

    Raises:
        LockError: If lock is already held by another process
    """
    existing = read_lock(session_dir)

    if existing is not None:
        if is_lock_stale(existing):
            # Stale lock - clear it and proceed
            release_lock(session_dir)
        else:
            # Active lock - cannot acquire
            raise LockError(
                f"Session locked by {existing.holder} (pid {existing.pid})",
                lock_info=existing,
            )

    # Create new lock
    lock_info = LockInfo(
        pid=os.getpid(),
        holder=holder,
        started=datetime.now(timezone.utc).isoformat(),
    )

    lock_path = get_lock_path(session_dir)
    with open(lock_path, "w") as f:
        yaml.safe_dump(lock_info.to_dict(), f)

    return lock_info


def release_lock(session_dir: Path) -> bool:
    """
    Release the lock on the session.

    Returns True if lock was released, False if no lock existed.
    Only releases if current process holds the lock.
    """
    lock_path = get_lock_path(session_dir)
    if not lock_path.exists():
        return False

    existing = read_lock(session_dir)
    if existing is None:
        # Invalid lock file - remove it
        lock_path.unlink()
        return True

    # Only release if we hold the lock (or it's stale)
    if existing.pid == os.getpid() or is_lock_stale(existing):
        lock_path.unlink()
        return True

    return False


def check_lock(session_dir: Path) -> Optional[LockInfo]:
    """
    Check if session is locked by another process.

    Returns LockInfo if locked by another process, None if available.
    Automatically clears stale locks.
    """
    existing = read_lock(session_dir)
    if existing is None:
        return None

    if is_lock_stale(existing):
        # Clear stale lock
        release_lock(session_dir)
        return None

    if existing.pid == os.getpid():
        # We hold the lock - session is available to us
        return None

    return existing


class SessionLock:
    """
    Context manager for session locking.

    Usage:
        with SessionLock(session_dir, "tui"):
            # Session is locked
            ...
        # Lock is released
    """

    def __init__(self, session_dir: Path, holder: str):
        self.session_dir = session_dir
        self.holder = holder
        self.lock_info: Optional[LockInfo] = None

    def __enter__(self) -> LockInfo:
        self.lock_info = acquire_lock(self.session_dir, self.holder)
        return self.lock_info

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        release_lock(self.session_dir)
        self.lock_info = None
