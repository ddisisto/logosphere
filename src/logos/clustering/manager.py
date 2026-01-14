"""
Cluster manager for persistence and session integration.

Handles saving/loading cluster state with efficient .npy storage for centroids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .models import (
    ClusterState,
    ClusterRegistry,
    AssignmentTable,
    ASSIGNMENT_NOISE,
    ASSIGNMENT_FOSSIL,
)
from .algorithm import (
    cosine_distance,
    process_iteration,
    bootstrap_clustering,
)


class ClusterManager:
    """
    Manages cluster state persistence and operations for a session.

    Storage format:
        session/clusters/
        ├── centroids.npy      # N × embedding_dim float32
        ├── meta.json          # cluster metadata + registry state
        └── assignments.json   # vector_id → cluster_id mapping
    """

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self._clusters_dir = self.session_dir / "clusters"
        self._centroids_path = self._clusters_dir / "centroids.npy"
        self._meta_path = self._clusters_dir / "meta.json"
        self._assignments_path = self._clusters_dir / "assignments.json"

        self.registry: Optional[ClusterRegistry] = None
        self.assignments: Optional[AssignmentTable] = None

        if self._meta_path.exists():
            self._load()

    @property
    def initialized(self) -> bool:
        """Check if clustering has been bootstrapped."""
        return self._meta_path.exists()

    def _load(self) -> None:
        """Load state from disk."""
        # Load centroids
        if self._centroids_path.exists():
            centroids = np.load(self._centroids_path)
        else:
            centroids = np.array([])

        # Load metadata
        with open(self._meta_path) as f:
            meta = json.load(f)

        # Reconstruct registry
        self.registry = ClusterRegistry(next_cluster_id=meta.get("next_cluster_id", 0))

        cluster_order = meta.get("cluster_order", [])
        clusters_meta = meta.get("clusters", {})

        for i, cluster_id in enumerate(cluster_order):
            if cluster_id in clusters_meta:
                centroid = centroids[i] if i < len(centroids) else np.zeros(1536, dtype=np.float32)
                self.registry.clusters[cluster_id] = ClusterState.from_meta_dict(
                    clusters_meta[cluster_id],
                    centroid
                )

        # Load assignments
        with open(self._assignments_path) as f:
            self.assignments = AssignmentTable.from_dict(json.load(f))

    def _save(self) -> None:
        """Save state to disk."""
        if self.registry is None or self.assignments is None:
            return

        self._clusters_dir.mkdir(parents=True, exist_ok=True)

        # Build ordered list of clusters for centroid array alignment
        cluster_order = list(self.registry.clusters.keys())

        # Save centroids as numpy array
        if cluster_order:
            centroids = np.array([
                self.registry.clusters[cid].centroid
                for cid in cluster_order
            ], dtype=np.float32)
            np.save(self._centroids_path, centroids)

        # Save metadata (without centroids)
        meta = {
            "next_cluster_id": self.registry.next_cluster_id,
            "cluster_order": cluster_order,
            "clusters": {
                cid: cluster.to_meta_dict()
                for cid, cluster in self.registry.clusters.items()
            },
        }
        with open(self._meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # Save assignments
        with open(self._assignments_path, 'w') as f:
            json.dump(self.assignments.to_dict(), f, indent=2)

    def bootstrap(
        self,
        session,
        min_cluster_size: int = 3,
        verbose: bool = True,
    ) -> dict:
        """Bootstrap clustering on existing messages."""
        self.registry = ClusterRegistry()
        self.assignments = AssignmentTable()

        stats = bootstrap_clustering(
            session,
            self.registry,
            self.assignments,
            min_cluster_size=min_cluster_size,
            verbose=verbose,
        )

        self._save()
        return stats

    def process(
        self,
        session,
        iteration: int,
        centroid_threshold: float = 0.3,
        min_cluster_size: int = 3,
        noise_window: int = 20,
        verbose: bool = True,
    ) -> dict:
        """Process one iteration of incremental clustering."""
        if not self.initialized:
            raise RuntimeError("Clustering not initialized. Run bootstrap first.")

        if self.registry is None:
            self._load()

        stats = process_iteration(
            session,
            self.registry,
            self.assignments,
            iteration,
            centroid_threshold=centroid_threshold,
            min_cluster_size=min_cluster_size,
            noise_window=noise_window,
            verbose=verbose,
        )

        self._save()
        return stats

    def get_status(self) -> dict:
        """Get clustering status summary."""
        if not self.initialized or self.registry is None:
            return {"initialized": False}

        # Count assignments by type
        cluster_counts = {}
        noise_count = 0
        fossil_count = 0

        for entry in self.assignments.assignments.values():
            if entry.cluster_id == ASSIGNMENT_NOISE:
                noise_count += 1
            elif entry.cluster_id == ASSIGNMENT_FOSSIL:
                fossil_count += 1
            else:
                cluster_counts[entry.cluster_id] = cluster_counts.get(entry.cluster_id, 0) + 1

        return {
            "initialized": True,
            "num_clusters": len(self.registry.clusters),
            "total_assigned": sum(cluster_counts.values()),
            "noise": noise_count,
            "fossil": fossil_count,
            "clusters": [
                {
                    "id": c.id,
                    "members": cluster_counts.get(c.id, 0),
                    "created_at": c.created_at,
                    "last_active": c.last_active,
                    "representative": c.representative_text[:80] + "..." if len(c.representative_text) > 80 else c.representative_text,
                }
                for c in sorted(self.registry.all_clusters(), key=lambda x: x.created_at)
            ],
        }

    def get_cluster_members(
        self,
        cluster_id: str,
        session,
        limit: int = 5,
    ) -> list[dict]:
        """
        Get detailed info for all members of a cluster.

        Args:
            cluster_id: The cluster to inspect
            session: Session object for vector_db access
            limit: Max members to return (0 = all)

        Returns:
            List of member dicts with metadata, distance, and text
        """
        if not self.initialized or self.registry is None:
            return []

        cluster = self.registry.get(cluster_id)
        if not cluster:
            return []

        vector_db = session.vector_db
        member_vids = self.assignments.get_cluster_members(cluster_id)

        members = []
        for vid in member_vids:
            meta = vector_db.get_message(vid)
            if not meta:
                continue

            embedding = vector_db.embeddings[vid]
            distance = cosine_distance(embedding, cluster.centroid)

            members.append({
                "vector_id": vid,
                "round": meta.get("round", 0),
                "mind_id": meta.get("mind_id", 0),
                "distance": float(distance),
                "text": meta.get("text", ""),
            })

        # Sort by distance (closest to centroid first)
        members.sort(key=lambda x: x["distance"])

        if limit > 0:
            members = members[:limit]

        return members
