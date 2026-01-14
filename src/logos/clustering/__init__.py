"""
Incremental clustering for Logos sessions.

Provides stable, persistent cluster assignments with evolving centroids.
See docs/incremental-clustering-design.md for full design rationale.
"""

from .models import (
    ClusterState,
    ClusterRegistry,
    AssignmentEntry,
    AssignmentTable,
    ASSIGNMENT_NOISE,
    ASSIGNMENT_FOSSIL,
)
from .algorithm import (
    cosine_distance,
    find_nearest_cluster,
    update_centroid,
    process_iteration,
    bootstrap_clustering,
    HDBSCAN_AVAILABLE,
)
from .manager import ClusterManager

__all__ = [
    # Models
    "ClusterState",
    "ClusterRegistry",
    "AssignmentEntry",
    "AssignmentTable",
    "ASSIGNMENT_NOISE",
    "ASSIGNMENT_FOSSIL",
    # Algorithm
    "cosine_distance",
    "find_nearest_cluster",
    "update_centroid",
    "process_iteration",
    "bootstrap_clustering",
    "HDBSCAN_AVAILABLE",
    # Manager
    "ClusterManager",
]
