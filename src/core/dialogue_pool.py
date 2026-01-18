"""
DialoguePool - Draft-based dialogue system for Logosphere v2.

Replaces MessagePool with a structured dialogue flow:
- User sends message → mind drafts responses → user accepts one
- Strict one-to-one: can't send new message until accepting a draft
- All drafts preserved with absolute indices (never pruned)

Storage:
    dialogue/
    └── pool.yaml    # awaiting, drafts, history
"""

from __future__ import annotations

import json
import tempfile
import shutil
from dataclasses import dataclass, field
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
    index: int  # Absolute index within this exchange (1, 2, 3, ...)
    iter: int
    time: str
    text: str
    seen: bool = False

    def to_dict(self) -> dict:
        return {
            'index': self.index,
            'iter': self.iter,
            'time': self.time,
            'text': self.text,
            'seen': self.seen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Draft:
        return cls(
            index=data.get('index', 0),  # Legacy compat
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
    accepted_draft_index: Optional[int] = None  # For mind entries: which draft was accepted
    draft_archive_id: Optional[str] = None  # For mind entries: exchange ID in draft archive

    def to_dict(self) -> dict:
        d = {'role': self.role, 'iter': self.iter, 'time': self.time, 'text': self.text}
        if self.accepted_draft_index is not None:
            d['accepted_draft_index'] = self.accepted_draft_index
        if self.draft_archive_id is not None:
            d['draft_archive_id'] = self.draft_archive_id
        return d

    @classmethod
    def from_dict(cls, data: dict) -> HistoryEntry:
        return cls(
            role=data['role'],
            iter=data['iter'],
            time=data['time'],
            text=data['text'],
            accepted_draft_index=data.get('accepted_draft_index'),
            draft_archive_id=data.get('draft_archive_id'),
        )


class DialoguePool:
    """
    Draft-based dialogue system.

    State machine:
    - IDLE: No awaiting message. History available for context.
    - DRAFTING: User message awaiting response. Drafts accumulate (never pruned).

    Flow:
    1. user sends message → state becomes DRAFTING
    2. mind iterations produce drafts (all preserved with absolute indices)
    3. user marks drafts as seen (optional)
    4. user accepts a draft → state becomes IDLE, exchange added to history

    Signal semantics:
    - opt-out (no draft): demands user attention to current buffer
    - short drafts: cycle buffer fast, can be used as signals
    - silence after good draft: implicit "ship it"
    """

    def __init__(
        self,
        pool_dir: Path,
    ):
        self.pool_dir = Path(pool_dir)

        # State
        self.awaiting: Optional[UserMessage] = None
        self.drafts: list[Draft] = []  # All drafts, oldest first, never pruned
        self.history: list[HistoryEntry] = []  # Oldest first, never pruned

        # Paths
        self._pool_path = self.pool_dir / 'pool.yaml'
        self._archive_path = self.pool_dir / 'draft_archive.jsonl'

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

    def add_draft(self, text: str, iter: int, seen: bool = False) -> int:
        """
        Mind produces a draft response.

        All drafts are preserved with absolute indices.

        Args:
            text: Draft content
            iter: Current iteration number
            seen: Mark as seen immediately (True when user is observing)

        Returns:
            The absolute index of the new draft
        """
        if not self.is_drafting:
            # No awaiting message - this shouldn't happen in normal flow
            return -1

        # Absolute index is 1-based, sequential
        index = len(self.drafts) + 1

        draft = Draft(
            index=index,
            iter=iter,
            time=datetime.now(timezone.utc).isoformat(),
            text=text,
            seen=seen,
        )

        self.drafts.append(draft)
        return index

    def mark_seen(self, indices: Optional[list[int]] = None) -> None:
        """
        Mark drafts as seen.

        Args:
            indices: Absolute indices (1, 2, 3...). None means mark all.
        """
        if indices is None:
            for draft in self.drafts:
                draft.seen = True
        else:
            for idx in indices:
                # Find draft with this absolute index
                for draft in self.drafts:
                    if draft.index == idx:
                        draft.seen = True
                        break

    def mark_all_unseen_as_seen(self) -> int:
        """Mark all currently unseen drafts as seen. Returns count marked."""
        count = 0
        for draft in self.drafts:
            if not draft.seen:
                draft.seen = True
                count += 1
        return count

    def accept(self, index: int) -> Draft:
        """
        Accept a draft by absolute index, moving the exchange to history.

        Args:
            index: Absolute index of draft to accept (1, 2, 3...)

        Returns:
            The accepted draft

        Raises:
            RuntimeError: If not in drafting state or no drafts
            IndexError: If index not found
        """
        if not self.is_drafting:
            raise RuntimeError("No pending message to accept draft for.")

        if not self.drafts:
            raise RuntimeError("No drafts available to accept.")

        # Find draft by absolute index
        accepted = None
        for draft in self.drafts:
            if draft.index == index:
                accepted = draft
                break

        if accepted is None:
            valid = [d.index for d in self.drafts]
            raise IndexError(f"Draft index {index} not found. Valid indices: {valid}")

        # Generate exchange ID and archive all drafts before clearing
        exchange_id = self._generate_exchange_id()
        self._archive_all_drafts(exchange_id, accepted.index)

        # Add to history: user message then accepted response (with draft index and archive ID)
        # History is never pruned - unlimited storage
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
            accepted_draft_index=accepted.index,
            draft_archive_id=exchange_id,
        ))

        # Clear drafting state
        self.awaiting = None
        self.drafts = []

        return accepted

    def get_latest_draft(self) -> Optional[Draft]:
        """Get the most recent draft, or None if no drafts."""
        return self.drafts[-1] if self.drafts else None

    def get_drafts_for_display(
        self,
        max_chars: int = 2000,
        max_count: int = 16,
    ) -> list[Draft]:
        """
        Get drafts for display to mind, newest first, within limits.

        Display stops when EITHER limit is reached:
        - max_chars: total character count
        - max_count: number of drafts

        Returns:
            List of drafts (newest first) that fit within limits
        """
        result = []
        total_chars = 0

        # Iterate newest first
        for draft in reversed(self.drafts):
            if len(result) >= max_count:
                break
            if total_chars + len(draft.text) > max_chars and result:
                # Would exceed char limit and we have at least one
                break
            result.append(draft)
            total_chars += len(draft.text)

        return result

    def get_all_drafts(self) -> list[Draft]:
        """Get all drafts (oldest first) for full display to user."""
        return list(self.drafts)

    def get_history(self) -> list[HistoryEntry]:
        """Get all conversation history (oldest first)."""
        return list(self.history)

    def get_history_for_display(
        self,
        max_entries: int,
        max_chars: Optional[int] = None,
    ) -> list[HistoryEntry]:
        """
        Get history for display to mind, within limits.

        Display stops when EITHER limit is reached:
        - max_entries: number of entries
        - max_chars: total character count

        Args:
            max_entries: Maximum number of entries to return
            max_chars: Optional maximum total characters

        Returns:
            List of history entries (oldest first within the returned subset)
        """
        # First apply count limit (take from end)
        if len(self.history) <= max_entries:
            candidates = list(self.history)
        else:
            candidates = list(self.history[-max_entries:])

        # If no char limit, return all candidates
        if max_chars is None:
            return candidates

        # Apply char limit (newest first, then reverse to oldest first)
        result = []
        total_chars = 0

        # Work backwards from newest to apply char limit
        for entry in reversed(candidates):
            if total_chars + len(entry.text) > max_chars and result:
                # Would exceed char limit and we have at least one
                break
            result.append(entry)
            total_chars += len(entry.text)

        # Reverse back to oldest-first order
        result.reverse()
        return result

    # -------------------------------------------------------------------------
    # Exchange ID and Draft Archive
    # -------------------------------------------------------------------------

    def _generate_exchange_id(self) -> str:
        """
        Generate a unique exchange ID for archiving drafts.

        Format: exc_{awaiting_iter}_{sequence:03d}

        The sequence number handles the (rare) case of multiple exchanges
        starting at the same iteration (e.g., after session restoration).
        """
        if self.awaiting is None:
            raise RuntimeError("Cannot generate exchange ID without awaiting message")

        base = f"exc_{self.awaiting.iter}"

        # Check existing exchanges to find the next sequence number
        sequence = 0
        if self._archive_path.exists():
            with open(self._archive_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        eid = entry.get('exchange_id', '')
                        if eid.startswith(base + '_'):
                            # Extract sequence number
                            suffix = eid[len(base) + 1:]
                            try:
                                seq = int(suffix)
                                sequence = max(sequence, seq + 1)
                            except ValueError:
                                pass
                    except json.JSONDecodeError:
                        pass

        return f"{base}_{sequence:03d}"

    def _archive_all_drafts(self, exchange_id: str, accepted_index: int) -> None:
        """
        Archive all current drafts to the JSONL archive file.

        Each entry contains:
        - exchange_id: The exchange this draft belongs to
        - draft_index: The absolute index of this draft within the exchange
        - iter_created: Iteration when draft was created
        - time_created: ISO timestamp when draft was created
        - text: The draft text content
        - user_seen: Whether the user had marked this draft as seen
        - accepted: Whether this draft was the accepted one
        - accepted_by_exchange: The exchange_id (redundant but useful for queries)
        """
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        with open(self._archive_path, 'a') as f:
            for draft in self.drafts:
                entry = {
                    'exchange_id': exchange_id,
                    'draft_index': draft.index,
                    'iter_created': draft.iter,
                    'time_created': draft.time,
                    'text': draft.text,
                    'user_seen': draft.seen,
                    'accepted': draft.index == accepted_index,
                    'accepted_by_exchange': exchange_id if draft.index == accepted_index else None,
                }
                f.write(json.dumps(entry) + '\n')

    def get_exchange_drafts(self, exchange_id: str) -> list[dict]:
        """
        Retrieve all archived drafts for a given exchange ID.

        Returns:
            List of draft entries (dicts) for the exchange, sorted by draft_index
        """
        if not self._archive_path.exists():
            return []

        results = []
        with open(self._archive_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('exchange_id') == exchange_id:
                        results.append(entry)
                except json.JSONDecodeError:
                    pass

        # Sort by draft_index
        results.sort(key=lambda x: x.get('draft_index', 0))
        return results

    def list_exchanges(self) -> list[dict]:
        """
        List all exchange IDs in the archive with metadata.

        Returns:
            List of dicts with:
            - exchange_id: The exchange ID
            - draft_count: Number of drafts in this exchange
            - accepted_index: Index of the accepted draft
            - first_iter: Iteration of first draft
            - last_iter: Iteration of last draft
        """
        if not self._archive_path.exists():
            return []

        exchanges: dict[str, dict] = {}

        with open(self._archive_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    eid = entry.get('exchange_id')
                    if not eid:
                        continue

                    if eid not in exchanges:
                        exchanges[eid] = {
                            'exchange_id': eid,
                            'draft_count': 0,
                            'accepted_index': None,
                            'first_iter': entry.get('iter_created'),
                            'last_iter': entry.get('iter_created'),
                        }

                    exchanges[eid]['draft_count'] += 1

                    if entry.get('accepted'):
                        exchanges[eid]['accepted_index'] = entry.get('draft_index')

                    iter_created = entry.get('iter_created')
                    if iter_created is not None:
                        if exchanges[eid]['first_iter'] is None or iter_created < exchanges[eid]['first_iter']:
                            exchanges[eid]['first_iter'] = iter_created
                        if exchanges[eid]['last_iter'] is None or iter_created > exchanges[eid]['last_iter']:
                            exchanges[eid]['last_iter'] = iter_created

                except json.JSONDecodeError:
                    pass

        # Sort by exchange_id (which implicitly sorts by creation order due to iter prefix)
        return sorted(exchanges.values(), key=lambda x: x['exchange_id'])

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
