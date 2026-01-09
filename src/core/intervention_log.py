"""
Intervention logging for Logosphere sessions.

Tracks all observer actions (inject, branch, rollback, run) with full context.
Append-only JSONL for auditability.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Intervention types
INTERVENTION_INJECT = "message_inject"
INTERVENTION_BRANCH = "branch"
INTERVENTION_ROLLBACK = "rollback"
INTERVENTION_RUN = "run"
INTERVENTION_SAVE = "save"


@dataclass
class Intervention:
    """Record of an observer action on the session."""

    id: str
    timestamp: str  # ISO format
    intervention_type: str
    snapshot_before: Optional[str]  # Snapshot ID before intervention
    snapshot_after: str  # Snapshot ID after intervention
    content: dict  # Type-specific payload
    notes: str

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Intervention":
        """Create from dict."""
        return cls(**data)

    def to_jsonl(self) -> str:
        """Convert to JSONL line."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_jsonl(cls, line: str) -> "Intervention":
        """Create from JSONL line."""
        return cls.from_dict(json.loads(line))


class InterventionLog:
    """
    Append-only log of session interventions.

    Stored as JSONL for streaming and crash-safety.
    """

    def __init__(self, log_path: Path):
        """
        Initialize intervention log.

        Args:
            log_path: Path to JSONL log file
        """
        self.log_path = Path(log_path)
        self._ensure_parent()

    def _ensure_parent(self) -> None:
        """Ensure parent directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        intervention_type: str,
        content: dict,
        snapshot_before: Optional[str],
        snapshot_after: str,
        notes: str = "",
    ) -> Intervention:
        """
        Record an intervention.

        Args:
            intervention_type: Type of intervention (use constants)
            content: Type-specific payload
            snapshot_before: Snapshot ID before intervention (None for init)
            snapshot_after: Snapshot ID after intervention
            notes: Optional observer notes

        Returns:
            Created Intervention record
        """
        intervention = Intervention(
            id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            intervention_type=intervention_type,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
            content=content,
            notes=notes,
        )

        # Append to log
        with open(self.log_path, 'a') as f:
            f.write(intervention.to_jsonl() + '\n')

        return intervention

    def query(
        self,
        intervention_type: Optional[str] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        snapshot_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Intervention]:
        """
        Query intervention history.

        Args:
            intervention_type: Filter by type
            after: Only interventions after this time
            before: Only interventions before this time
            snapshot_id: Only interventions involving this snapshot
            limit: Maximum results (newest first)

        Returns:
            List of matching Interventions (newest first)
        """
        if not self.log_path.exists():
            return []

        results = []

        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                intervention = Intervention.from_jsonl(line)

                # Apply filters
                if intervention_type and intervention.intervention_type != intervention_type:
                    continue

                if after or before:
                    ts = datetime.fromisoformat(intervention.timestamp)
                    if after and ts <= after:
                        continue
                    if before and ts >= before:
                        continue

                if snapshot_id:
                    if (intervention.snapshot_before != snapshot_id and
                            intervention.snapshot_after != snapshot_id):
                        continue

                results.append(intervention)

        # Newest first
        results.reverse()

        if limit:
            results = results[:limit]

        return results

    def all(self) -> list[Intervention]:
        """Get all interventions (oldest first)."""
        if not self.log_path.exists():
            return []

        results = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(Intervention.from_jsonl(line))

        return results

    def latest(self) -> Optional[Intervention]:
        """Get most recent intervention."""
        results = self.query(limit=1)
        return results[0] if results else None
