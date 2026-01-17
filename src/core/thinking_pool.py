"""
ThinkingPool - Embedded thought storage for Logosphere v2.

Stores thoughts with embeddings for clustering and similarity search.
Uses YAML for human-readable metadata, .npy for efficient embedding storage.

Key differences from v1 VectorDB:
- YAML metadata (pool.yaml) instead of JSONL
- Cluster info included in thought metadata
- FIFO rotation for active pool
"""

from __future__ import annotations

import random
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
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


class Thought:
    """A single thought in the thinking pool."""

    __slots__ = ('text', 'embedding', 'iter', 'time', 'cluster', 'vector_id')

    def __init__(
        self,
        text: str,
        embedding: np.ndarray,
        iter: int,
        time: str,
        cluster: Optional[str] = None,
        vector_id: Optional[int] = None,
    ):
        self.text = text
        self.embedding = embedding.astype(np.float32)
        self.iter = iter
        self.time = time
        self.cluster = cluster  # None until assigned, then "cluster_N" or "noise" or "fossil"
        self.vector_id = vector_id

    def to_dict(self) -> dict:
        """Convert to YAML-serializable dict."""
        return {
            'text': self.text,
            'iter': self.iter,
            'time': self.time,
            'cluster': self.cluster,
            'vector_id': self.vector_id,
        }

    @classmethod
    def from_dict(cls, data: dict, embedding: np.ndarray) -> Thought:
        """Create from dict + embedding."""
        return cls(
            text=data['text'],
            embedding=embedding,
            iter=data['iter'],
            time=data['time'],
            cluster=data.get('cluster'),
            vector_id=data.get('vector_id'),
        )


class ThinkingPool:
    """
    Embedded thought storage with clustering support.

    Storage format:
        thinking/
        ├── embeddings.npy    # (N, embedding_dim) float32
        └── pool.yaml         # List of thought metadata

    Active pool = tail N thoughts (FIFO rotation).
    """

    def __init__(
        self,
        pool_dir: Path,
        active_pool_size: int = 50,
        embedding_dim: int = 1536,
    ):
        self.pool_dir = Path(pool_dir)
        self.active_pool_size = active_pool_size
        self.embedding_dim = embedding_dim

        # Storage
        self.thoughts: list[Thought] = []

        # Paths
        self._embeddings_path = self.pool_dir / 'embeddings.npy'
        self._pool_path = self.pool_dir / 'pool.yaml'

        # Load if exists
        if self._pool_path.exists():
            self._load()

    def add(
        self,
        text: str,
        embedding: np.ndarray,
        iter: int,
        cluster: Optional[str] = None,
    ) -> int:
        """
        Add a thought to the pool.

        Args:
            text: Thought content
            embedding: Vector embedding
            iter: Iteration number when created
            cluster: Optional initial cluster assignment

        Returns:
            vector_id of the new thought
        """
        vector_id = len(self.thoughts)

        thought = Thought(
            text=text,
            embedding=embedding,
            iter=iter,
            time=datetime.now(timezone.utc).isoformat(),
            cluster=cluster,
            vector_id=vector_id,
        )

        self.thoughts.append(thought)
        return vector_id

    def sample(self, k: int) -> tuple[list[Thought], list[int]]:
        """
        Sample k thoughts from active pool (FIFO tail).

        Returns:
            (thoughts, vector_ids)
        """
        indices = self._active_indices()
        if not indices:
            return [], []

        sample_size = min(k, len(indices))
        sampled_indices = random.sample(indices, sample_size)

        thoughts = [self.thoughts[i] for i in sampled_indices]
        return thoughts, sampled_indices

    def get(self, vector_id: int) -> Optional[Thought]:
        """Get thought by vector_id."""
        if 0 <= vector_id < len(self.thoughts):
            return self.thoughts[vector_id]
        return None

    def update_cluster(self, vector_id: int, cluster: str) -> None:
        """Update cluster assignment for a thought."""
        if 0 <= vector_id < len(self.thoughts):
            self.thoughts[vector_id].cluster = cluster

    def get_embedding(self, vector_id: int) -> Optional[np.ndarray]:
        """Get embedding by vector_id."""
        if 0 <= vector_id < len(self.thoughts):
            return self.thoughts[vector_id].embedding
        return None

    def get_all_embeddings(self) -> np.ndarray:
        """Get all embeddings as numpy array."""
        if not self.thoughts:
            return np.array([]).reshape(0, self.embedding_dim)
        return np.array([t.embedding for t in self.thoughts])

    def size(self) -> int:
        """Total thoughts in pool."""
        return len(self.thoughts)

    def active_size(self) -> int:
        """Size of active pool."""
        return len(self._active_indices())

    def _active_indices(self) -> list[int]:
        """Get indices of active pool (FIFO tail)."""
        total = len(self.thoughts)
        if total <= self.active_pool_size:
            return list(range(total))
        return list(range(total - self.active_pool_size, total))

    def get_visible_ids(self) -> set[int]:
        """Get all vector IDs (for clustering compatibility)."""
        return set(range(len(self.thoughts)))

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self) -> None:
        """Save pool to disk with atomic writes."""
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        if self.thoughts:
            embeddings = np.array([t.embedding for t in self.thoughts])
            np.save(self._embeddings_path, embeddings)

        # Save metadata as YAML (atomic write)
        pool_data = [t.to_dict() for t in self.thoughts]

        with tempfile.NamedTemporaryFile(
            mode='w', dir=self.pool_dir, suffix='.tmp', delete=False
        ) as f:
            temp_path = Path(f.name)
            yaml.dump(pool_data, f, Dumper=_LiteralDumper,
                      default_flow_style=False, allow_unicode=True)

        shutil.move(str(temp_path), str(self._pool_path))

    def _load(self) -> None:
        """Load pool from disk."""
        # Load metadata
        with open(self._pool_path) as f:
            pool_data = yaml.safe_load(f) or []

        # Load embeddings
        if self._embeddings_path.exists() and pool_data:
            embeddings = np.load(self._embeddings_path)
            self.embedding_dim = embeddings.shape[1] if len(embeddings) > 0 else self.embedding_dim
        else:
            embeddings = np.array([]).reshape(0, self.embedding_dim)

        # Reconstruct thoughts
        self.thoughts = []
        for i, data in enumerate(pool_data):
            emb = embeddings[i] if i < len(embeddings) else np.zeros(self.embedding_dim)
            thought = Thought.from_dict(data, emb)
            self.thoughts.append(thought)

    # -------------------------------------------------------------------------
    # Clustering interface (compatible with existing clustering code)
    # -------------------------------------------------------------------------

    def get_message(self, vector_id: int) -> Optional[dict]:
        """Get message dict for clustering compatibility."""
        thought = self.get(vector_id)
        if thought:
            return {
                'text': thought.text,
                'round': thought.iter,  # Clustering uses 'round'
                'vector_id': thought.vector_id,
            }
        return None
