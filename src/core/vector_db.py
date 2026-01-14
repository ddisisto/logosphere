"""
VectorDB - Unified message storage with embeddings for Logosphere.

Replaces Pool as single source of truth. Stores messages, embeddings,
and metadata. Supports random sampling (like Pool) plus similarity search.

SCALING NOTE: Uses sklearn NearestNeighbors (brute-force cosine). This is
fine for M=200 active pool (~10ms search). If scaling to 10k+ vectors,
swap to faiss-cpu or hnswlib - the interface is designed for drop-in replacement.
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


class VectorDB:
    """
    Unified storage for messages + embeddings.

    Replaces Pool with added vector search capabilities.
    Active pool = tail M messages (same semantics as Pool).
    """

    def __init__(self, active_pool_size: int, embedding_dim: int = 1536):
        """
        Initialize VectorDB.

        Args:
            active_pool_size: M (tail size for active pool)
            embedding_dim: Embedding vector dimension (1536 for text-embedding-3-small)
        """
        self.active_pool_size = active_pool_size
        self.embedding_dim = embedding_dim

        # Storage
        self.embeddings: list[np.ndarray] = []  # One per message
        self.metadata: list[dict] = []  # One per message

        # Index (rebuilt on search, lazy)
        self._index: Optional[NearestNeighbors] = None
        self._index_dirty = True

    def add(
        self,
        text: str,
        embedding: np.ndarray,
        round_num: int,
        mind_id: int,
        extra_metadata: Optional[dict] = None
    ) -> int:
        """
        Add message with embedding.

        Args:
            text: Message content
            embedding: Vector embedding (1536-dim)
            round_num: Experiment round number
            mind_id: Which Mind produced this message
            extra_metadata: Optional additional metadata

        Returns:
            vector_id: Index of added message
        """
        vector_id = len(self.embeddings)

        # Build metadata with consistent field order for readability
        # (text last since it's longest and variable-length)
        meta = {
            'round': round_num,
            'mind_id': mind_id,
            'vector_id': vector_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        if extra_metadata:
            meta.update(extra_metadata)
        meta['text'] = text  # Always last for readability

        self.embeddings.append(embedding.astype(np.float32))
        self.metadata.append(meta)
        self._index_dirty = True

        return vector_id

    def sample_random(self, k: int, from_active_pool: bool = True) -> list[str]:
        """
        Sample k messages uniformly at random.

        Drop-in replacement for Pool.sample().

        Args:
            k: Number of messages to sample
            from_active_pool: If True, sample from tail M only

        Returns:
            List of message texts
        """
        if from_active_pool:
            indices = self._active_indices()
        else:
            indices = list(range(len(self.metadata)))

        if len(indices) == 0:
            return []

        sample_size = min(k, len(indices))
        sampled_indices = random.sample(indices, sample_size)

        return [self.metadata[i]['text'] for i in sampled_indices]

    def sample_with_ids(
        self, k: int, from_active_pool: bool = True
    ) -> tuple[list[str], list[int]]:
        """
        Sample k messages and return both texts and vector_ids.

        Args:
            k: Number of messages to sample
            from_active_pool: If True, sample from tail M only

        Returns:
            Tuple of (texts, vector_ids)
        """
        if from_active_pool:
            indices = self._active_indices()
        else:
            indices = list(range(len(self.metadata)))

        if len(indices) == 0:
            return [], []

        sample_size = min(k, len(indices))
        sampled_indices = random.sample(indices, sample_size)

        texts = [self.metadata[i]['text'] for i in sampled_indices]
        ids = [self.metadata[i]['vector_id'] for i in sampled_indices]
        return texts, ids

    def sample_weighted(self, k: int, weights: dict[int, float]) -> list[str]:
        """
        Weighted sampling from active pool.

        For intervention strategies that boost/reduce certain messages.

        Args:
            weights: {vector_id: weight} - higher = more likely to sample
                     Missing IDs get weight 1.0

        Returns:
            List of message texts
        """
        indices = self._active_indices()
        if len(indices) == 0:
            return []

        # Build weight array
        w = np.array([weights.get(i, 1.0) for i in indices])
        w = w / w.sum()  # Normalize to probabilities

        sample_size = min(k, len(indices))
        sampled_indices = np.random.choice(
            indices, size=sample_size, replace=False, p=w
        )

        return [self.metadata[i]['text'] for i in sampled_indices]

    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        from_active_pool: bool = False
    ) -> list[dict]:
        """
        Find k most similar messages by cosine similarity.

        Args:
            query_embedding: Query vector
            k: Number of results
            from_active_pool: Search only active pool if True

        Returns:
            List of {vector_id, distance, text, metadata}
        """
        if len(self.embeddings) == 0:
            return []

        if from_active_pool:
            indices = self._active_indices()
            if len(indices) == 0:
                return []
            embeddings = np.array([self.embeddings[i] for i in indices])
            id_map = indices
        else:
            embeddings = np.array(self.embeddings)
            id_map = list(range(len(self.embeddings)))

        # Rebuild index if needed
        # SCALING: For 10k+ vectors, swap to faiss.IndexFlatIP or hnswlib
        index = NearestNeighbors(
            n_neighbors=min(k, len(embeddings)),
            metric='cosine',
            algorithm='brute'  # Fine for <10k vectors
        )
        index.fit(embeddings)

        distances, local_indices = index.kneighbors(
            query_embedding.reshape(1, -1)
        )

        results = []
        for dist, local_idx in zip(distances[0], local_indices[0]):
            global_idx = id_map[local_idx]
            results.append({
                'vector_id': global_idx,
                'distance': float(dist),
                'text': self.metadata[global_idx]['text'],
                'metadata': self.metadata[global_idx],
            })

        return results

    def get_active_pool_data(self) -> tuple[np.ndarray, list[dict]]:
        """
        Get active pool embeddings and metadata for clustering.

        Returns:
            (embeddings array [M x dim], metadata list)
        """
        indices = self._active_indices()
        if len(indices) == 0:
            return np.array([]).reshape(0, self.embedding_dim), []

        embeddings = np.array([self.embeddings[i] for i in indices])
        metadata = [self.metadata[i] for i in indices]

        return embeddings, metadata

    def get_message(self, vector_id: int) -> Optional[dict]:
        """Get metadata for a specific message."""
        if 0 <= vector_id < len(self.metadata):
            return self.metadata[vector_id]
        return None

    def size(self) -> int:
        """Total messages in history."""
        return len(self.metadata)

    def active_size(self) -> int:
        """Current active pool size."""
        return len(self._active_indices())

    def _active_indices(self) -> list[int]:
        """Get indices of active pool (tail M messages)."""
        total = len(self.metadata)
        if total <= self.active_pool_size:
            return list(range(total))
        return list(range(total - self.active_pool_size, total))

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save to disk using atomic writes to prevent corruption from concurrent access.

        Creates:
            path/embeddings.npy - Embedding matrix
            path/metadata.jsonl - One JSON object per message
        """
        import tempfile
        import shutil

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings as numpy array (atomic via numpy's temp file handling)
        if self.embeddings:
            np.save(path / 'embeddings.npy', np.array(self.embeddings))

        # Save metadata as JSONL using atomic write (temp file + rename)
        metadata_path = path / 'metadata.jsonl'
        with tempfile.NamedTemporaryFile(mode='w', dir=path, suffix='.tmp', delete=False) as f:
            temp_path = Path(f.name)
            for meta in self.metadata:
                f.write(json.dumps(meta) + '\n')

        # Atomic rename (on POSIX systems)
        shutil.move(str(temp_path), str(metadata_path))

    @classmethod
    def load(cls, path: Path, active_pool_size: int) -> 'VectorDB':
        """
        Load from disk.

        Args:
            path: Directory containing embeddings.npy and metadata.jsonl
            active_pool_size: M for the loaded VectorDB

        Returns:
            Loaded VectorDB instance
        """
        path = Path(path)

        # Load metadata first to get count
        metadata = []
        with open(path / 'metadata.jsonl', 'r') as f:
            for line in f:
                metadata.append(json.loads(line))

        # Infer embedding dim from file
        embeddings_path = path / 'embeddings.npy'
        if embeddings_path.exists() and len(metadata) > 0:
            embeddings = np.load(embeddings_path)
            embedding_dim = embeddings.shape[1]
        else:
            embeddings = np.array([])
            embedding_dim = 1536

        # Create instance
        db = cls(active_pool_size=active_pool_size, embedding_dim=embedding_dim)
        db.metadata = metadata
        db.embeddings = list(embeddings) if len(embeddings) > 0 else []

        return db

    @classmethod
    def load_from_legacy(
        cls,
        npz_path: Path,
        jsonl_path: Path,
        active_pool_size: int
    ) -> 'VectorDB':
        """
        Load from existing baseline experiment format.

        For validating against existing .npz embeddings + experiment.jsonl logs.

        Args:
            npz_path: Path to embeddings.npz (from analyze.py embeddings tool)
            jsonl_path: Path to experiment.jsonl logs
            active_pool_size: M for the loaded VectorDB

        Returns:
            VectorDB populated with existing experiment data
        """
        # Load embeddings
        data = np.load(npz_path, allow_pickle=True)
        embeddings = data['embeddings']
        texts = data['messages']  # Legacy format uses 'messages' key

        # Create instance
        db = cls(
            active_pool_size=active_pool_size,
            embedding_dim=embeddings.shape[1]
        )

        # Parse JSONL to extract round/mind info per message
        # Build text -> (round, mind_id) mapping from logs
        text_to_meta = {}
        with open(jsonl_path, 'r') as f:
            for line in f:
                event = json.loads(line)
                if event.get('event') == 'mind_invocation':
                    round_num = event.get('round', 0)
                    mind_id = event.get('mind_id', 0)
                    for msg in event.get('transmitted', []):
                        text_to_meta[msg] = (round_num, mind_id)

        # Add messages in order
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            round_num, mind_id = text_to_meta.get(text, (0, 0))
            db.embeddings.append(emb.astype(np.float32))
            db.metadata.append({
                'vector_id': i,
                'text': text,
                'round': round_num,
                'mind_id': mind_id,
                'timestamp': None,  # Not available in legacy format
            })

        return db
