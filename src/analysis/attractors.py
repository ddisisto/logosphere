"""
Attractor detection via HDBSCAN clustering.

Identifies semantic clusters (attractors) in the active pool.
Used for real-time detection during experiments and post-hoc analysis.
"""

from typing import Optional

import numpy as np

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from ..core.vector_db import VectorDB


class AttractorDetector:
    """
    Detect attractors via density-based clustering on active pool embeddings.

    Uses HDBSCAN which:
    - Auto-determines cluster count
    - Handles noise (not all points forced into clusters)
    - Finds natural density-based clusters
    """

    def __init__(
        self,
        vector_db: VectorDB,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
    ):
        """
        Initialize detector.

        Args:
            vector_db: VectorDB instance to cluster
            min_cluster_size: Minimum points to form a cluster (default 5)
            min_samples: Core point threshold (default: same as min_cluster_size)
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "hdbscan not installed. Run: uv sync --extra analysis && pip install hdbscan"
            )

        self.vector_db = vector_db
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size

        # Cache last detection result
        self._last_state: Optional[dict] = None

    def detect(self, round_num: int) -> dict:
        """
        Cluster active pool embeddings and return attractor state.

        Args:
            round_num: Current experiment round (for logging)

        Returns:
            {
                'detected': bool,           # True if any clusters found
                'round': int,               # Round number
                'num_clusters': int,        # Number of clusters (excluding noise)
                'noise_count': int,         # Points not in any cluster
                'clusters': [               # Per-cluster info
                    {
                        'id': int,          # Cluster label
                        'size': int,        # Number of messages
                        'coherence': float, # Intra-cluster similarity (0-1)
                        'centroid': list,   # Mean embedding (as list for JSON)
                        'representative': str,  # Message closest to centroid
                        'rounds': list[int] # When members were created
                    },
                    ...
                ]
            }
        """
        embeddings, metadata = self.vector_db.get_active_pool_data()

        if len(embeddings) < self.min_cluster_size:
            return self._empty_state(round_num)

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',  # Works well with normalized embeddings
        )
        labels = clusterer.fit_predict(embeddings)

        # Build cluster info
        clusters = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        for label in sorted(unique_labels):
            mask = labels == label
            cluster_embeddings = embeddings[mask]
            cluster_metadata = [m for m, is_member in zip(metadata, mask) if is_member]

            # Centroid
            centroid = cluster_embeddings.mean(axis=0)

            # Coherence: mean pairwise cosine similarity within cluster
            coherence = self._compute_coherence(cluster_embeddings)

            # Representative: message closest to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            rep_idx = np.argmin(distances)
            representative = cluster_metadata[rep_idx]['text']

            # Rounds when members were created
            rounds = sorted(set(m['round'] for m in cluster_metadata))

            clusters.append({
                'id': int(label),
                'size': int(mask.sum()),
                'coherence': float(coherence),
                'centroid': centroid.tolist(),
                'representative': representative,
                'rounds': rounds,
            })

        # Sort by size descending
        clusters.sort(key=lambda c: c['size'], reverse=True)

        noise_count = int((labels == -1).sum())

        state = {
            'detected': len(clusters) > 0,
            'round': round_num,
            'num_clusters': len(clusters),
            'noise_count': noise_count,
            'clusters': clusters,
        }

        self._last_state = state
        return state

    def get_state(self) -> Optional[dict]:
        """Return last detection state (for intervention hooks)."""
        return self._last_state

    def _compute_coherence(self, embeddings: np.ndarray) -> float:
        """
        Compute intra-cluster coherence as mean pairwise cosine similarity.

        Returns value in [0, 1] where 1 = all identical.
        """
        if len(embeddings) < 2:
            return 1.0

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)

        # Mean pairwise similarity (excluding self-similarity)
        similarities = normalized @ normalized.T
        n = len(embeddings)

        # Sum off-diagonal and normalize
        total = similarities.sum() - n  # Subtract diagonal (all 1s)
        pairs = n * (n - 1)

        return float(total / pairs) if pairs > 0 else 1.0

    def _empty_state(self, round_num: int) -> dict:
        """Return empty state when not enough data."""
        return {
            'detected': False,
            'round': round_num,
            'num_clusters': 0,
            'noise_count': 0,
            'clusters': [],
        }


def analyze_attractors(vector_db: VectorDB, min_cluster_size: int = 5) -> dict:
    """
    Convenience function for post-hoc analysis.

    Args:
        vector_db: Loaded VectorDB with experiment data
        min_cluster_size: Minimum cluster size

    Returns:
        Attractor state dict
    """
    detector = AttractorDetector(
        vector_db=vector_db,
        min_cluster_size=min_cluster_size,
    )
    return detector.detect(round_num=-1)  # -1 indicates post-hoc
