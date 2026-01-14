"""
Clustering algorithms for incremental clustering.

Core functions for two-phase clustering: centroid matching + HDBSCAN discovery.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from .models import (
    ClusterState,
    ClusterRegistry,
    AssignmentTable,
    ASSIGNMENT_NOISE,
)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
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
        distance = cosine_distance(embedding, cluster.centroid)
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
            if cosine_distance(embedding, best_cluster.centroid) < cosine_distance(rep_embedding, best_cluster.centroid):
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
