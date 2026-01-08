"""
Snapshot management for Logosphere sessions.

Provides save/load/fork operations on VectorDB state with lineage tracking.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .vector_db import VectorDB


def _slugify(text: str, max_length: int = 40) -> str:
    """Convert text to URL-safe slug."""
    # Lowercase, replace spaces/special chars with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower())
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    # Truncate
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit('-', 1)[0]
    return slug or 'snapshot'


def _generate_snapshot_id(description: str) -> str:
    """Generate human-readable snapshot ID from description + timestamp."""
    slug = _slugify(description)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    short_uuid = uuid.uuid4().hex[:6]
    return f"{slug}-{timestamp}-{short_uuid}"


@dataclass
class Snapshot:
    """Immutable record of VectorDB state at a point in time."""

    id: str
    created_at: str  # ISO format
    parent_id: Optional[str]
    description: str
    vector_db_path: str  # Relative to snapshots dir
    iteration: int  # Current iteration number
    config: dict
    metrics: dict  # Diversity, clusters at snapshot time

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Snapshot":
        """Create from dict."""
        return cls(**data)


class SnapshotStore:
    """
    Manages snapshot storage and lineage tracking.

    Storage layout:
        base_dir/
        ├── index.json          # Snapshot metadata and lineage
        └── <snapshot_id>/
            ├── vector_db/      # VectorDB.save() output
            ├── config.json
            └── metrics.json
    """

    def __init__(self, base_dir: Path):
        """
        Initialize snapshot store.

        Args:
            base_dir: Directory for snapshot storage
        """
        self.base_dir = Path(base_dir)
        self._index_path = self.base_dir / "index.json"
        self._index: dict = {"snapshots": [], "lineage": {}}

        # Load existing index if present
        if self._index_path.exists():
            self._load_index()

    def _load_index(self) -> None:
        """Load index from disk."""
        with open(self._index_path) as f:
            self._index = json.load(f)

    def _save_index(self) -> None:
        """Save index to disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, 'w') as f:
            json.dump(self._index, f, indent=2)

    def save(
        self,
        vector_db: VectorDB,
        description: str,
        iteration: int,
        parent_id: Optional[str] = None,
        config: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ) -> Snapshot:
        """
        Save VectorDB state as a new snapshot.

        Args:
            vector_db: VectorDB to snapshot
            description: Human-readable description
            iteration: Current iteration number
            parent_id: Parent snapshot ID (for lineage tracking)
            config: Configuration at snapshot time
            metrics: Pool metrics at snapshot time

        Returns:
            Created Snapshot
        """
        snapshot_id = _generate_snapshot_id(description)
        snapshot_dir = self.base_dir / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save VectorDB
        vector_db_path = snapshot_dir / "vector_db"
        vector_db.save(vector_db_path)

        # Save config
        config = config or {}
        with open(snapshot_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Save metrics
        metrics = metrics or {}
        with open(snapshot_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create snapshot record
        snapshot = Snapshot(
            id=snapshot_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            parent_id=parent_id,
            description=description,
            vector_db_path=str(vector_db_path.relative_to(self.base_dir)),
            iteration=iteration,
            config=config,
            metrics=metrics,
        )

        # Update index
        self._index["snapshots"].append(snapshot.to_dict())
        self._index["lineage"][snapshot_id] = parent_id
        self._save_index()

        return snapshot

    def load(self, snapshot_id: str, active_pool_size: int = 50) -> VectorDB:
        """
        Load VectorDB from a snapshot.

        Args:
            snapshot_id: Snapshot ID to load
            active_pool_size: Active pool size for loaded VectorDB

        Returns:
            Loaded VectorDB

        Raises:
            FileNotFoundError: If snapshot doesn't exist
        """
        snapshot = self.get(snapshot_id)
        if snapshot is None:
            raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")

        vector_db_path = self.base_dir / snapshot.vector_db_path
        return VectorDB.load(vector_db_path, active_pool_size=active_pool_size)

    def get(self, snapshot_id: str) -> Optional[Snapshot]:
        """
        Get snapshot metadata by ID.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            Snapshot or None if not found
        """
        for snap_dict in self._index["snapshots"]:
            if snap_dict["id"] == snapshot_id:
                return Snapshot.from_dict(snap_dict)
        return None

    def list(self, lineage_root: Optional[str] = None) -> list[Snapshot]:
        """
        List all snapshots, optionally filtered by lineage.

        Args:
            lineage_root: If provided, only return snapshots descended from this ID

        Returns:
            List of Snapshots (newest first)
        """
        snapshots = [Snapshot.from_dict(s) for s in self._index["snapshots"]]

        if lineage_root:
            # Filter to descendants of lineage_root
            descendants = self._get_descendants(lineage_root)
            snapshots = [s for s in snapshots if s.id in descendants or s.id == lineage_root]

        # Sort by created_at descending
        snapshots.sort(key=lambda s: s.created_at, reverse=True)
        return snapshots

    def get_lineage(self, snapshot_id: str) -> list[Snapshot]:
        """
        Get ancestry chain from root to given snapshot.

        Args:
            snapshot_id: Snapshot to trace ancestry from

        Returns:
            List of Snapshots from oldest ancestor to given snapshot
        """
        chain = []
        current_id = snapshot_id

        while current_id:
            snapshot = self.get(current_id)
            if snapshot is None:
                break
            chain.append(snapshot)
            current_id = self._index["lineage"].get(current_id)

        chain.reverse()  # Oldest first
        return chain

    def _get_descendants(self, root_id: str) -> set[str]:
        """Get all snapshot IDs descended from root."""
        descendants = set()

        # Build parent -> children mapping
        children_map: dict[str, list[str]] = {}
        for snap_id, parent_id in self._index["lineage"].items():
            if parent_id:
                children_map.setdefault(parent_id, []).append(snap_id)

        # BFS from root
        queue = [root_id]
        while queue:
            current = queue.pop(0)
            for child in children_map.get(current, []):
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)

        return descendants

    def latest(self) -> Optional[Snapshot]:
        """Get the most recent snapshot."""
        if not self._index["snapshots"]:
            return None
        # Snapshots are appended in order, so last is newest
        return Snapshot.from_dict(self._index["snapshots"][-1])
