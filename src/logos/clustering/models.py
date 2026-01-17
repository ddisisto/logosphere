"""
Data models for incremental clustering.

Defines the core data structures for cluster state and assignments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Assignment states
ASSIGNMENT_NOISE = "noise"


@dataclass
class ClusterState:
    """State of a single cluster."""

    id: str                      # e.g., "cluster_0"
    centroid: np.ndarray         # Current centroid (evolves with new members)
    member_count: int            # Total assigned messages
    created_at: int              # Iteration when spawned
    last_active: int             # Last iteration with new assignment
    representative_id: int       # vector_id of message closest to centroid
    representative_text: str     # Text of representative message

    def to_meta_dict(self) -> dict:
        """Convert to dict for JSON serialization (excludes centroid)."""
        return {
            "id": self.id,
            "member_count": self.member_count,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "representative_id": self.representative_id,
            "representative_text": self.representative_text,
        }

    @classmethod
    def from_meta_dict(cls, data: dict, centroid: np.ndarray) -> ClusterState:
        """Create from metadata dict + centroid array."""
        return cls(
            id=data["id"],
            centroid=centroid,
            member_count=data["member_count"],
            created_at=data["created_at"],
            last_active=data["last_active"],
            representative_id=data["representative_id"],
            representative_text=data["representative_text"],
        )


@dataclass
class ClusterRegistry:
    """Registry of all clusters and their states."""

    clusters: dict[str, ClusterState] = field(default_factory=dict)
    next_cluster_id: int = 0

    def get(self, cluster_id: str) -> Optional[ClusterState]:
        return self.clusters.get(cluster_id)

    def all_clusters(self) -> list[ClusterState]:
        return list(self.clusters.values())

    def spawn_cluster(
        self,
        centroid: np.ndarray,
        iteration: int,
        representative_id: int,
        representative_text: str,
    ) -> ClusterState:
        """Create a new cluster."""
        cluster_id = f"cluster_{self.next_cluster_id}"
        self.next_cluster_id += 1

        cluster = ClusterState(
            id=cluster_id,
            centroid=centroid.astype(np.float32),
            member_count=0,
            created_at=iteration,
            last_active=iteration,
            representative_id=representative_id,
            representative_text=representative_text,
        )
        self.clusters[cluster_id] = cluster
        return cluster


@dataclass
class AssignmentEntry:
    """Assignment record for a single message."""

    vector_id: int
    cluster_id: str              # cluster_id or "noise"
    assigned_at: int             # Iteration when assigned

    def to_dict(self) -> dict:
        return {
            "vector_id": self.vector_id,
            "cluster_id": self.cluster_id,
            "assigned_at": self.assigned_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AssignmentEntry:
        return cls(
            vector_id=data["vector_id"],
            cluster_id=data["cluster_id"],
            assigned_at=data["assigned_at"],
        )


class AssignmentTable:
    """Tracks message -> cluster assignments."""

    def __init__(self):
        self.assignments: dict[int, AssignmentEntry] = {}  # vector_id -> entry

    def get(self, vector_id: int) -> Optional[AssignmentEntry]:
        return self.assignments.get(vector_id)

    def assign(self, vector_id: int, cluster_id: str, iteration: int) -> None:
        """Assign a message to a cluster."""
        self.assignments[vector_id] = AssignmentEntry(
            vector_id=vector_id,
            cluster_id=cluster_id,
            assigned_at=iteration,
        )

    def mark_noise(self, vector_id: int, iteration: int) -> None:
        """Mark a message as noise."""
        self.assignments[vector_id] = AssignmentEntry(
            vector_id=vector_id,
            cluster_id=ASSIGNMENT_NOISE,
            assigned_at=iteration,
        )

    def get_all_noise(self) -> list[int]:
        """Get all vector_ids currently marked as noise."""
        return [
            vid for vid, entry in self.assignments.items()
            if entry.cluster_id == ASSIGNMENT_NOISE
        ]

    def get_unassigned(self, vector_ids: set[int]) -> list[int]:
        """Get vector_ids that have no assignment."""
        return [vid for vid in vector_ids if vid not in self.assignments]

    def get_cluster_members(self, cluster_id: str) -> list[int]:
        """Get all vector_ids assigned to a cluster."""
        return [
            vid for vid, entry in self.assignments.items()
            if entry.cluster_id == cluster_id
        ]

    def to_dict(self) -> dict:
        return {
            "assignments": {
                str(k): v.to_dict() for k, v in self.assignments.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> AssignmentTable:
        table = cls()
        for k, v in data.get("assignments", {}).items():
            table.assignments[int(k)] = AssignmentEntry.from_dict(v)
        return table
