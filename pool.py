"""
Message pool - storage and sampling for Logosphere experiment.

The pool maintains full message history but sampling occurs only from
the active pool (tail M messages).
"""

import random


class Pool:
    def __init__(self, max_active: int):
        """
        Initialize message pool.

        Args:
            max_active: M (tail size for active pool sampling)
        """
        self.max_active = max_active
        self.messages: list[str] = []

    def add_message(self, content: str) -> None:
        """Add message to pool (appends to history)."""
        self.messages.append(content)

    def sample(self, k: int) -> list[str]:
        """
        Sample k messages uniformly at random from active pool (tail M).

        If fewer than k messages in active pool, returns all available.

        Args:
            k: Number of messages to sample

        Returns:
            List of k sampled messages (or fewer if pool too small)
        """
        active = self.get_active()

        if len(active) == 0:
            return []

        # Sample min(k, len(active)) messages without replacement
        sample_size = min(k, len(active))
        return random.sample(active, sample_size)

    def get_active(self) -> list[str]:
        """Return active pool (tail M messages)."""
        if len(self.messages) <= self.max_active:
            return self.messages.copy()
        return self.messages[-self.max_active:]

    def get_all(self) -> list[str]:
        """Return full message history."""
        return self.messages.copy()

    def size(self) -> int:
        """Return total messages in history."""
        return len(self.messages)

    def active_size(self) -> int:
        """Return current active pool size."""
        return len(self.get_active())
