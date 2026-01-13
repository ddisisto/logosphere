"""
Incremental clustering for Logos sessions.

Provides stable, persistent cluster assignments with evolving centroids.
See docs/incremental-clustering-design.md for full design rationale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# Assignment states
ASSIGNMENT_NOISE = "noise"
ASSIGNMENT_FOSSIL = "fossil"


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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "centroid": self.centroid.tolist(),
            "member_count": self.member_count,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "representative_id": self.representative_id,
            "representative_text": self.representative_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ClusterState:
        return cls(
            id=data["id"],
            centroid=np.array(data["centroid"], dtype=np.float32),
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

    def to_dict(self) -> dict:
        return {
            "clusters": {k: v.to_dict() for k, v in self.clusters.items()},
            "next_cluster_id": self.next_cluster_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ClusterRegistry:
        registry = cls(next_cluster_id=data.get("next_cluster_id", 0))
        for k, v in data.get("clusters", {}).items():
            registry.clusters[k] = ClusterState.from_dict(v)
        return registry


@dataclass
class AssignmentEntry:
    """Assignment record for a single message."""

    vector_id: int
    cluster_id: str              # cluster_id, "noise", or "fossil"
    assigned_at: int             # Iteration when assigned
    noise_since: Optional[int]   # If noise, when it became noise (for aging)

    def to_dict(self) -> dict:
        d = {
            "vector_id": self.vector_id,
            "cluster_id": self.cluster_id,
            "assigned_at": self.assigned_at,
        }
        if self.noise_since is not None:
            d["noise_since"] = self.noise_since
        return d

    @classmethod
    def from_dict(cls, data: dict) -> AssignmentEntry:
        return cls(
            vector_id=data["vector_id"],
            cluster_id=data["cluster_id"],
            assigned_at=data["assigned_at"],
            noise_since=data.get("noise_since"),
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
            noise_since=None,
        )

    def mark_noise(self, vector_id: int, iteration: int) -> None:
        """Mark a message as noise."""
        self.assignments[vector_id] = AssignmentEntry(
            vector_id=vector_id,
            cluster_id=ASSIGNMENT_NOISE,
            assigned_at=iteration,
            noise_since=iteration,
        )

    def mark_fossil(self, vector_id: int, iteration: int) -> None:
        """Mark noise as fossilized (no longer reconsidered)."""
        entry = self.assignments.get(vector_id)
        if entry and entry.cluster_id == ASSIGNMENT_NOISE:
            entry.cluster_id = ASSIGNMENT_FOSSIL
            entry.assigned_at = iteration

    def get_recent_noise(self, current_iteration: int, window: int) -> list[int]:
        """Get vector_ids of noise from the last N iterations."""
        cutoff = current_iteration - window
        return [
            vid for vid, entry in self.assignments.items()
            if entry.cluster_id == ASSIGNMENT_NOISE
            and entry.noise_since is not None
            and entry.noise_since >= cutoff
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


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 1.0
    return 1.0 - np.dot(a, b) / (norm_a * norm_b)


def find_nearest_cluster(
    embedding: np.ndarray,
    registry: ClusterRegistry,
) -> tuple[Optional[ClusterState], float]:
    """Find the nearest cluster to an embedding."""
    best_cluster = None
    best_distance = float('inf')

    for cluster in registry.all_clusters():
        distance = _cosine_distance(embedding, cluster.centroid)
        if distance < best_distance:
            best_distance = distance
            best_cluster = cluster

    return best_cluster, best_distance


def update_centroid(cluster: ClusterState, new_embedding: np.ndarray, iteration: int) -> None:
    """Update cluster centroid with a new member."""
    n = cluster.member_count
    cluster.centroid = (cluster.centroid * n + new_embedding) / (n + 1)
    cluster.member_count += 1
    cluster.last_active = iteration


def process_iteration(
    session,  # Session object
    registry: ClusterRegistry,
    assignments: AssignmentTable,
    iteration: int,
    centroid_threshold: float = 0.3,
    min_cluster_size: int = 3,
    noise_window: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Process one iteration of incremental clustering.

    Two-phase algorithm:
    1. Match new messages + recent noise against existing cluster centroids
    2. Run HDBSCAN on remaining unmatched to discover new clusters

    Args:
        session: Session object with vector_db access
        registry: ClusterRegistry to update
        assignments: AssignmentTable to update
        iteration: Current iteration number
        centroid_threshold: Max cosine distance for phase 1 matching
        min_cluster_size: HDBSCAN threshold for new cluster formation
        noise_window: How long noise stays in candidate pool (iterations)
        verbose: Print progress

    Returns:
        Dict with stats: {assigned, new_clusters, noise, fossilized}
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan not installed. Run: uv sync --extra analysis")

    vector_db = session.vector_db
    visible_ids = session.get_visible_ids()

    # 1. Gather candidates: unassigned messages + recent noise
    unassigned = assignments.get_unassigned(visible_ids)
    recent_noise = assignments.get_recent_noise(iteration, noise_window)
    candidates = list(set(unassigned + recent_noise))

    if verbose and candidates:
        print(f"  Processing {len(unassigned)} new + {len(recent_noise)} noise = {len(candidates)} candidates")

    stats = {"assigned": 0, "new_clusters": 0, "noise": 0, "fossilized": 0}

    if not candidates:
        return stats

    # Gather embeddings for candidates
    candidate_embeddings = {}
    candidate_metadata = {}
    for vid in candidates:
        meta = vector_db.get_message(vid)
        if meta:
            candidate_embeddings[vid] = vector_db.embeddings[vid]
            candidate_metadata[vid] = meta

    # 2. Phase 1: Match against existing clusters
    unmatched = []
    for vid in candidates:
        if vid not in candidate_embeddings:
            continue

        embedding = candidate_embeddings[vid]
        best_cluster, distance = find_nearest_cluster(embedding, registry)

        if best_cluster and distance < centroid_threshold:
            # Assign to existing cluster
            assignments.assign(vid, best_cluster.id, iteration)
            update_centroid(best_cluster, embedding, iteration)

            # Update representative if this is closer to centroid
            rep_embedding = vector_db.embeddings[best_cluster.representative_id]
            if _cosine_distance(embedding, best_cluster.centroid) < _cosine_distance(rep_embedding, best_cluster.centroid):
                best_cluster.representative_id = vid
                best_cluster.representative_text = candidate_metadata[vid]['text']

            stats["assigned"] += 1
        else:
            unmatched.append(vid)

    # 3. Phase 2: Discover new clusters from unmatched
    if len(unmatched) >= min_cluster_size:
        unmatched_embeddings = np.array([candidate_embeddings[vid] for vid in unmatched])

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_cluster_size,
            metric='euclidean',
        )
        labels = clusterer.fit_predict(unmatched_embeddings)

        # Group by label
        label_to_vids: dict[int, list[int]] = {}
        for vid, label in zip(unmatched, labels):
            if label not in label_to_vids:
                label_to_vids[label] = []
            label_to_vids[label].append(vid)

        # Spawn new clusters
        for label, vids in label_to_vids.items():
            if label == -1:
                continue  # noise handled below

            # Compute centroid
            embeddings = np.array([candidate_embeddings[vid] for vid in vids])
            centroid = embeddings.mean(axis=0)

            # Find representative (closest to centroid)
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            rep_idx = np.argmin(distances)
            rep_vid = vids[rep_idx]
            rep_text = candidate_metadata[rep_vid]['text']

            # Spawn cluster
            cluster = registry.spawn_cluster(
                centroid=centroid,
                iteration=iteration,
                representative_id=rep_vid,
                representative_text=rep_text,
            )

            # Assign all members
            for vid in vids:
                assignments.assign(vid, cluster.id, iteration)
                update_centroid(cluster, candidate_embeddings[vid], iteration)

            stats["new_clusters"] += 1
            stats["assigned"] += len(vids)

            # Remove from unmatched
            for vid in vids:
                unmatched.remove(vid)

    # 4. Mark remaining as noise
    for vid in unmatched:
        existing = assignments.get(vid)
        if existing and existing.cluster_id == ASSIGNMENT_NOISE:
            # Already noise, check if should fossilize
            if existing.noise_since and (iteration - existing.noise_since) >= noise_window:
                assignments.mark_fossil(vid, iteration)
                stats["fossilized"] += 1
            # else: stays as noise, will be reconsidered
        else:
            # New noise
            assignments.mark_noise(vid, iteration)
            stats["noise"] += 1

    if verbose:
        print(f"  Assigned: {stats['assigned']}, New clusters: {stats['new_clusters']}, "
              f"Noise: {stats['noise']}, Fossilized: {stats['fossilized']}")

    return stats


def bootstrap_clustering(
    session,
    registry: ClusterRegistry,
    assignments: AssignmentTable,
    min_cluster_size: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Bootstrap clustering on all existing messages.

    Runs HDBSCAN on all visible messages to establish initial clusters.

    Returns:
        Dict with stats: {clusters, assigned, noise}
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan not installed. Run: uv sync --extra analysis")

    vector_db = session.vector_db
    visible_ids = sorted(session.get_visible_ids())

    if verbose:
        print(f"Bootstrapping clusters from {len(visible_ids)} messages...")

    if len(visible_ids) < min_cluster_size:
        if verbose:
            print(f"Not enough messages for clustering (need {min_cluster_size})")
        return {"clusters": 0, "assigned": 0, "noise": len(visible_ids)}

    # Gather all embeddings
    embeddings = []
    metadata = []
    for vid in visible_ids:
        meta = vector_db.get_message(vid)
        if meta:
            embeddings.append(vector_db.embeddings[vid])
            metadata.append(meta)

    embeddings = np.array(embeddings)

    # Run HDBSCAN
    if verbose:
        print(f"Running HDBSCAN with min_cluster_size={min_cluster_size}...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_cluster_size,
        metric='euclidean',
    )
    labels = clusterer.fit_predict(embeddings)

    # Group by label
    label_to_indices: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)

    stats = {"clusters": 0, "assigned": 0, "noise": 0}

    # Create clusters
    for label, indices in label_to_indices.items():
        if label == -1:
            # Noise
            for i in indices:
                vid = visible_ids[i]
                round_num = metadata[i].get('round', 0)
                assignments.mark_noise(vid, round_num)
                stats["noise"] += 1
            continue

        # Compute centroid
        cluster_embeddings = embeddings[indices]
        centroid = cluster_embeddings.mean(axis=0)

        # Find representative
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        rep_local_idx = np.argmin(distances)
        rep_idx = indices[rep_local_idx]
        rep_vid = visible_ids[rep_idx]
        rep_text = metadata[rep_idx]['text']

        # Get iteration from earliest message in cluster
        min_round = min(metadata[i].get('round', 0) for i in indices)

        # Spawn cluster
        cluster = registry.spawn_cluster(
            centroid=centroid,
            iteration=min_round,
            representative_id=rep_vid,
            representative_text=rep_text,
        )

        # Assign all members
        for i in indices:
            vid = visible_ids[i]
            round_num = metadata[i].get('round', 0)
            assignments.assign(vid, cluster.id, round_num)
            cluster.member_count += 1

        # Update last_active to most recent message
        max_round = max(metadata[i].get('round', 0) for i in indices)
        cluster.last_active = max_round

        stats["clusters"] += 1
        stats["assigned"] += len(indices)

    if verbose:
        print(f"Created {stats['clusters']} clusters, assigned {stats['assigned']} messages, "
              f"{stats['noise']} noise")

    return stats


class ClusterManager:
    """
    Manages cluster state persistence and operations for a session.
    """

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self._registry_path = self.session_dir / "clusters.json"
        self._assignments_path = self.session_dir / "assignments.json"

        self.registry: Optional[ClusterRegistry] = None
        self.assignments: Optional[AssignmentTable] = None

        if self._registry_path.exists():
            self._load()

    @property
    def initialized(self) -> bool:
        """Check if clustering has been bootstrapped."""
        return self._registry_path.exists()

    def _load(self) -> None:
        """Load state from disk."""
        with open(self._registry_path) as f:
            self.registry = ClusterRegistry.from_dict(json.load(f))

        with open(self._assignments_path) as f:
            self.assignments = AssignmentTable.from_dict(json.load(f))

    def _save(self) -> None:
        """Save state to disk."""
        if self.registry is None or self.assignments is None:
            return

        with open(self._registry_path, 'w') as f:
            json.dump(self.registry.to_dict(), f, indent=2)

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
