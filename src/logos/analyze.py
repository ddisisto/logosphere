"""
Cluster timeline analysis for Logos sessions.

Retroactively computes cluster evolution with identity tracking via centroid matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


@dataclass
class TrackedCluster:
    """A cluster tracked across iterations."""

    id: str                      # Stable ID (e.g., "cluster_0")
    sizes: list[int]             # Size at each iteration (0 if not present)
    first_seen: int              # First iteration where cluster appeared
    last_seen: int               # Last iteration where cluster appeared
    centroid: np.ndarray         # Current centroid for matching
    coherence: float             # Current intra-cluster similarity
    representative: str          # Message closest to centroid


@dataclass
class ClusterTimeline:
    """Timeline of cluster evolution across iterations."""

    iterations: list[int]
    clusters: list[TrackedCluster]
    total_messages: int = 0

    def to_json(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "iterations": self.iterations,
            "total_messages": self.total_messages,
            "clusters": [
                {
                    "id": c.id,
                    "sizes": c.sizes,
                    "first_seen": c.first_seen,
                    "last_seen": c.last_seen,
                    "coherence": c.coherence,
                    "representative": c.representative,
                }
                for c in self.clusters
            ],
            "summary": self._compute_summary(),
        }

    def _compute_summary(self) -> dict:
        """Compute summary statistics."""
        if not self.clusters:
            return {
                "total_clusters_seen": 0,
                "dominant_cluster": None,
                "phase_transitions": [],
            }

        # Find dominant cluster (largest final size)
        final_sizes = [(c.id, c.sizes[-1] if c.sizes else 0) for c in self.clusters]
        dominant = max(final_sizes, key=lambda x: x[1])

        # Detect phase transitions (dominant cluster changes)
        transitions = []
        if len(self.iterations) > 1:
            prev_dominant = None
            for i, iteration in enumerate(self.iterations):
                sizes_at_iter = [(c.id, c.sizes[i]) for c in self.clusters]
                current_dominant = max(sizes_at_iter, key=lambda x: x[1])[0] if sizes_at_iter else None

                if prev_dominant and current_dominant != prev_dominant:
                    transitions.append({
                        "iteration": iteration,
                        "from": prev_dominant,
                        "to": current_dominant,
                    })
                prev_dominant = current_dominant

        return {
            "total_clusters_seen": len(self.clusters),
            "dominant_cluster": dominant[0] if dominant[1] > 0 else None,
            "phase_transitions": transitions,
        }

    def to_swimlane_ascii(self, width: int = 40) -> str:
        """Generate ASCII swimlane visualization."""
        if not self.iterations or not self.clusters:
            return "No cluster data available."

        lines = []
        min_iter = min(self.iterations)
        max_iter = max(self.iterations)
        iter_range = max_iter - min_iter + 1

        lines.append(f"Cluster Timeline (iterations {min_iter}-{max_iter})")
        lines.append("")

        # Sort clusters by first_seen
        sorted_clusters = sorted(self.clusters, key=lambda c: c.first_seen)

        # Find max label width for alignment
        max_label_len = max(len(f"{c.id} ({sum(c.sizes):3d})") for c in sorted_clusters) if sorted_clusters else 0

        for cluster in sorted_clusters:
            total_msgs = sum(c for c in cluster.sizes if c > 0)
            label = f"{cluster.id} ({total_msgs:3d})"
            label = label.ljust(max_label_len)

            # Build bar
            bar = []
            max_size = max(max(c.sizes) for c in self.clusters) if self.clusters else 1

            for i, size in enumerate(cluster.sizes):
                if size == 0:
                    bar.append("░")
                elif size < max_size * 0.25:
                    bar.append("▒")
                elif size < max_size * 0.5:
                    bar.append("▓")
                else:
                    bar.append("█")

            # Scale bar to width
            if len(bar) > width:
                # Downsample
                scaled_bar = []
                for j in range(width):
                    idx = int(j * len(bar) / width)
                    scaled_bar.append(bar[idx])
                bar = scaled_bar
            elif len(bar) < width:
                # Pad
                bar = bar + ["░"] * (width - len(bar))

            bar_str = "".join(bar)

            # Status annotation
            if cluster.last_seen < max_iter:
                status = f"  died at {cluster.last_seen}"
            elif cluster.sizes[-1] == max(c.sizes[-1] for c in self.clusters):
                status = f"  dominant, coh={cluster.coherence:.2f}"
            else:
                status = ""

            lines.append(f"{label}: {bar_str}{status}")

        # Add axis
        lines.append("")
        axis_label = " " * (max_label_len + 2)
        axis_ticks = []
        for i in range(0, width + 1, width // 4):
            iter_val = min_iter + int(i * iter_range / width)
            axis_ticks.append(f"{iter_val}")

        # Build axis line
        axis_line = axis_label + "|"
        tick_positions = [0, width // 4, width // 2, 3 * width // 4, width]
        for i, pos in enumerate(tick_positions[:-1]):
            next_pos = tick_positions[i + 1]
            segment = "-" * (next_pos - pos - 1) + "+"
            axis_line += segment
        lines.append(axis_line[:-1] + "|")

        # Tick labels
        tick_labels = axis_label
        for i, pos in enumerate(tick_positions):
            iter_val = min_iter + int(pos * iter_range / width) if iter_range > 0 else min_iter
            tick_labels += f"{iter_val:<{width // 4}}" if i < len(tick_positions) - 1 else str(iter_val)
        lines.append(tick_labels[:len(axis_line)])

        # Summary
        summary = self._compute_summary()
        lines.append("")
        lines.append("Summary:")
        lines.append(f"  Total clusters seen: {summary['total_clusters_seen']}")
        if summary['dominant_cluster']:
            dominant = next((c for c in self.clusters if c.id == summary['dominant_cluster']), None)
            if dominant and self.total_messages > 0:
                pct = dominant.sizes[-1] / self.total_messages * 100
                lines.append(f"  Current dominant: {summary['dominant_cluster']} ({pct:.0f}% of pool)")
        for trans in summary['phase_transitions']:
            lines.append(f"  Phase transition: {trans['from']} → {trans['to']} at iteration {trans['iteration']}")

        return "\n".join(lines)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 1.0
    return 1.0 - np.dot(a, b) / (norm_a * norm_b)


def _compute_coherence(embeddings: np.ndarray) -> float:
    """Compute intra-cluster coherence as mean pairwise cosine similarity."""
    if len(embeddings) < 2:
        return 1.0

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    similarities = normalized @ normalized.T
    n = len(embeddings)
    total = similarities.sum() - n
    pairs = n * (n - 1)
    return float(total / pairs) if pairs > 0 else 1.0


def compute_cluster_timeline(
    session,  # Session object (avoid circular import)
    min_cluster_size: int = 3,
    centroid_match_threshold: float = 0.3,
    verbose: bool = True,
) -> ClusterTimeline:
    """
    Retroactively compute cluster evolution for a session.

    Algorithm:
    1. Get max_iteration from session
    2. For each iteration i in 0..max_iteration:
       a. Filter messages to those with round <= i
       b. Get embeddings for those messages
       c. Run HDBSCAN clustering
       d. Match clusters to previous iteration by centroid proximity
       e. Track sizes, births, deaths
    3. Return ClusterTimeline

    Args:
        session: Logos Session object
        min_cluster_size: Minimum points to form a cluster
        centroid_match_threshold: Max cosine distance to consider clusters "same"
        verbose: Print progress

    Returns:
        ClusterTimeline with tracked clusters
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan not installed. Run: pip install hdbscan")

    vector_db = session.vector_db
    max_iteration = session.iteration

    if verbose:
        print(f"Computing cluster timeline for iterations 0-{max_iteration}...")

    # Get all message data
    all_embeddings = []
    all_metadata = []
    for vid in range(vector_db.size()):
        meta = vector_db.get_message(vid)
        if meta:
            all_metadata.append(meta)
            all_embeddings.append(vector_db.embeddings[vid])

    if not all_embeddings:
        return ClusterTimeline(iterations=[], clusters=[], total_messages=0)

    all_embeddings = np.array(all_embeddings)

    # Track clusters across iterations
    tracked_clusters: dict[str, TrackedCluster] = {}
    next_cluster_id = 0
    iterations = list(range(max_iteration + 1))

    # Previous iteration's cluster centroids for matching
    prev_centroids: dict[str, np.ndarray] = {}

    for iteration in iterations:
        # Filter to messages at or before this iteration
        mask = np.array([m.get('round', 0) <= iteration for m in all_metadata])
        iter_embeddings = all_embeddings[mask]
        iter_metadata = [m for m, keep in zip(all_metadata, mask) if keep]

        if len(iter_embeddings) < min_cluster_size:
            # Not enough data - record zero sizes for all tracked clusters
            for cluster in tracked_clusters.values():
                cluster.sizes.append(0)
            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration}: {len(iter_embeddings)} msgs, 0 clusters")
            continue

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_cluster_size,
            metric='euclidean',
        )
        labels = clusterer.fit_predict(iter_embeddings)

        # Extract current clusters
        current_clusters: dict[int, dict] = {}
        unique_labels = set(labels)
        unique_labels.discard(-1)

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_embeddings = iter_embeddings[cluster_mask]
            cluster_metadata = [m for m, is_member in zip(iter_metadata, cluster_mask) if is_member]

            centroid = cluster_embeddings.mean(axis=0)
            coherence = _compute_coherence(cluster_embeddings)

            # Find representative (closest to centroid)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            rep_idx = np.argmin(distances)
            representative = cluster_metadata[rep_idx]['text']

            current_clusters[label] = {
                'centroid': centroid,
                'size': int(cluster_mask.sum()),
                'coherence': coherence,
                'representative': representative,
            }

        # Match to previous iteration's clusters
        matched_current: set[int] = set()
        matched_tracked: set[str] = set()

        for label, cluster_info in current_clusters.items():
            best_match = None
            best_distance = centroid_match_threshold

            for tracked_id, prev_centroid in prev_centroids.items():
                if tracked_id in matched_tracked:
                    continue
                dist = _cosine_distance(cluster_info['centroid'], prev_centroid)
                if dist < best_distance:
                    best_distance = dist
                    best_match = tracked_id

            if best_match:
                # Update existing cluster
                tracked_clusters[best_match].sizes.append(cluster_info['size'])
                tracked_clusters[best_match].last_seen = iteration
                tracked_clusters[best_match].centroid = cluster_info['centroid']
                tracked_clusters[best_match].coherence = cluster_info['coherence']
                tracked_clusters[best_match].representative = cluster_info['representative']
                matched_current.add(label)
                matched_tracked.add(best_match)

        # Create new clusters for unmatched
        for label, cluster_info in current_clusters.items():
            if label not in matched_current:
                new_id = f"cluster_{next_cluster_id}"
                next_cluster_id += 1

                # Backfill sizes with 0 for previous iterations
                sizes = [0] * iteration + [cluster_info['size']]

                tracked_clusters[new_id] = TrackedCluster(
                    id=new_id,
                    sizes=sizes,
                    first_seen=iteration,
                    last_seen=iteration,
                    centroid=cluster_info['centroid'],
                    coherence=cluster_info['coherence'],
                    representative=cluster_info['representative'],
                )

        # Record zero for clusters not seen this iteration
        for tracked_id, cluster in tracked_clusters.items():
            if tracked_id not in matched_tracked and len(cluster.sizes) <= iteration:
                cluster.sizes.append(0)

        # Update prev_centroids for next iteration (keep all clusters, not just alive ones)
        prev_centroids = {
            tracked_id: cluster.centroid
            for tracked_id, cluster in tracked_clusters.items()
        }

        if verbose and iteration % 10 == 0:
            print(f"  Iteration {iteration}: {len(iter_embeddings)} msgs, {len(current_clusters)} clusters")

    # Ensure all clusters have sizes for all iterations
    for cluster in tracked_clusters.values():
        while len(cluster.sizes) < len(iterations):
            cluster.sizes.append(0)

    # Sort clusters by first_seen
    clusters = sorted(tracked_clusters.values(), key=lambda c: c.first_seen)

    # Compute total messages at final iteration
    total_messages = sum(1 for m in all_metadata if m.get('round', 0) <= max_iteration)

    if verbose:
        print(f"Done. Found {len(clusters)} clusters across {len(iterations)} iterations.")

    return ClusterTimeline(
        iterations=iterations,
        clusters=clusters,
        total_messages=total_messages,
    )
