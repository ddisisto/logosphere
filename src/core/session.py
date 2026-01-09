"""
Session management for Logosphere.

Branch-based model: single append-only VectorDB with branch views.
No snapshots - branches are just filters over the global ID space.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .vector_db import VectorDB
from .embedding_client import EmbeddingClient
from .intervention_log import (
    InterventionLog,
    Intervention,
    INTERVENTION_INJECT,
    INTERVENTION_BRANCH,
    INTERVENTION_RUN,
)


@dataclass
class Branch:
    """A branch is a view over the VectorDB."""

    name: str
    parent: Optional[str]  # Parent branch name
    parent_iteration: Optional[int]  # Iteration we branched at
    id_ranges: list[list]  # [[start, end], ...] - end=None means open

    def contains(self, vector_id: int) -> bool:
        """Check if this branch contains the given vector_id."""
        for start, end in self.id_ranges:
            if end is None:
                if vector_id >= start:
                    return True
            elif start <= vector_id <= end:
                return True
        return False

    def add_id(self, vector_id: int) -> None:
        """Add a vector_id to this branch (extends last open range or creates new)."""
        if self.id_ranges and self.id_ranges[-1][1] is None:
            # Last range is open, ID should be consecutive
            pass  # Nothing to do, open range includes it
        else:
            # Start new open range
            self.id_ranges.append([vector_id, None])

    def close_range(self, last_id: int) -> None:
        """Close the current open range at last_id."""
        if self.id_ranges and self.id_ranges[-1][1] is None:
            self.id_ranges[-1][1] = last_id

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parent": self.parent,
            "parent_iteration": self.parent_iteration,
            "id_ranges": self.id_ranges,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Branch:
        return cls(**data)


class Session:
    """
    Manages a Logosphere session with branch-based history.

    Single append-only VectorDB. Branches are views (filters) over IDs.
    """

    def __init__(
        self,
        session_dir: Path,
        active_pool_size: int = 50,
        embedding_dim: int = 1536,
    ):
        self.session_dir = Path(session_dir)
        self.active_pool_size = active_pool_size
        self.embedding_dim = embedding_dim

        # Paths
        self._state_path = self.session_dir / "state.json"
        self._branches_path = self.session_dir / "branches.json"
        self._vector_db_path = self.session_dir / "vector_db"

        # State
        self.vector_db: Optional[VectorDB] = None
        self.branches: dict[str, Branch] = {}
        self.current_branch: str = "main"
        self.iteration: int = 0
        self.config: dict = {}

        # Intervention log (optional, for audit)
        self.intervention_log = InterventionLog(self.session_dir / "interventions.jsonl")

        # Load or init
        if self._branches_path.exists():
            self._load()
        else:
            self._init()

    def _init(self) -> None:
        """Initialize fresh session."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.vector_db = VectorDB(
            active_pool_size=self.active_pool_size,
            embedding_dim=self.embedding_dim
        )

        # Create main branch with open range starting at 0
        self.branches = {
            "main": Branch(
                name="main",
                parent=None,
                parent_iteration=None,
                id_ranges=[[0, None]]
            )
        }
        self.current_branch = "main"
        self.iteration = 0
        self.config = {}

        self._save()

    def _load(self) -> None:
        """Load session from disk."""
        # Load branches
        with open(self._branches_path) as f:
            data = json.load(f)

        self.current_branch = data["current"]
        self.iteration = data["iteration"]
        self.config = data.get("config", {})
        self.branches = {
            name: Branch.from_dict(b)
            for name, b in data["branches"].items()
        }

        # Load VectorDB
        if self._vector_db_path.exists():
            self.vector_db = VectorDB.load(
                self._vector_db_path,
                active_pool_size=self.active_pool_size
            )
        else:
            self.vector_db = VectorDB(
                active_pool_size=self.active_pool_size,
                embedding_dim=self.embedding_dim
            )

    def _save(self) -> None:
        """Save session to disk."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Save branches
        data = {
            "current": self.current_branch,
            "iteration": self.iteration,
            "config": self.config,
            "branches": {
                name: b.to_dict()
                for name, b in self.branches.items()
            }
        }
        with open(self._branches_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Save VectorDB
        self.vector_db.save(self._vector_db_path)

    @property
    def current(self) -> Branch:
        """Current branch object."""
        return self.branches[self.current_branch]

    def get_visible_ids(self) -> set[int]:
        """Get all vector_ids visible to current branch."""
        visible = set()
        branch = self.current

        # Add own IDs
        for start, end in branch.id_ranges:
            if end is None:
                end = self.vector_db.size() - 1
            for i in range(start, end + 1):
                visible.add(i)

        # Add inherited IDs from parent (up to branch point)
        if branch.parent and branch.parent_iteration is not None:
            parent_branch = self.branches.get(branch.parent)
            if parent_branch:
                visible.update(self._get_inherited_ids(parent_branch, branch.parent_iteration))

        return visible

    def _get_inherited_ids(self, branch: Branch, up_to_iteration: int) -> set[int]:
        """Get IDs from branch up to given iteration."""
        visible = set()

        for start, end in branch.id_ranges:
            if end is None:
                end = self.vector_db.size() - 1
            for vid in range(start, end + 1):
                meta = self.vector_db.get_message(vid)
                if meta and meta.get('round', 0) <= up_to_iteration:
                    visible.add(vid)

        # Recurse to parent
        if branch.parent and branch.parent_iteration is not None:
            parent_branch = self.branches.get(branch.parent)
            if parent_branch:
                # Use the earlier of the two branch points
                effective_iter = min(up_to_iteration, branch.parent_iteration)
                visible.update(self._get_inherited_ids(parent_branch, effective_iter))

        return visible

    def sample(self, k: int) -> tuple[list[str], list[int]]:
        """
        Sample k messages from current branch's visible pool.

        Returns:
            (texts, vector_ids)
        """
        visible_ids = list(self.get_visible_ids())
        if not visible_ids:
            return [], []

        # Get active pool (tail of visible IDs)
        visible_ids.sort()
        if len(visible_ids) > self.active_pool_size:
            active_ids = visible_ids[-self.active_pool_size:]
        else:
            active_ids = visible_ids

        # Sample
        import random
        sample_size = min(k, len(active_ids))
        sampled_ids = random.sample(active_ids, sample_size)

        texts = [self.vector_db.get_message(vid)['text'] for vid in sampled_ids]
        return texts, sampled_ids

    def add(
        self,
        text: str,
        embedding,
        mind_id: int = 0,
        extra_metadata: Optional[dict] = None,
    ) -> int:
        """Add message to VectorDB and current branch."""
        vector_id = self.vector_db.add(
            text=text,
            embedding=embedding,
            round_num=self.iteration,
            mind_id=mind_id,
            extra_metadata=extra_metadata,
        )

        # Branch automatically includes it (open range)
        return vector_id

    def branch(self, name: str) -> str:
        """
        Create new branch from current state.

        Args:
            name: New branch name

        Returns:
            Name of created branch
        """
        if name in self.branches:
            raise ValueError(f"Branch '{name}' already exists")

        # Close current branch's range
        current_last_id = self.vector_db.size() - 1
        if current_last_id >= 0:
            self.current.close_range(current_last_id)

        # Create new branch
        new_branch = Branch(
            name=name,
            parent=self.current_branch,
            parent_iteration=self.iteration,
            id_ranges=[[self.vector_db.size(), None]]  # New IDs start here
        )
        self.branches[name] = new_branch

        # Log intervention
        self.intervention_log.record(
            intervention_type=INTERVENTION_BRANCH,
            content={
                "from_branch": self.current_branch,
                "to_branch": name,
                "at_iteration": self.iteration,
            },
            snapshot_before=self.current_branch,
            snapshot_after=name,
        )

        # Switch to new branch
        self.current_branch = name

        self._save()
        return name

    def switch(self, name: str) -> None:
        """
        Switch to existing branch.

        Args:
            name: Branch name to switch to
        """
        if name not in self.branches:
            raise ValueError(f"Branch '{name}' not found")

        # Close current branch's range if it has open range
        current_last_id = self.vector_db.size() - 1
        if current_last_id >= 0 and self.current.id_ranges:
            last_range = self.current.id_ranges[-1]
            if last_range[1] is None:
                # Check if we actually added any IDs to this range
                if current_last_id >= last_range[0]:
                    self.current.close_range(current_last_id)

        # Switch
        self.current_branch = name

        # Reopen target branch's range for new additions
        target = self.branches[name]
        target.id_ranges.append([self.vector_db.size(), None])

        self._save()

    def inject_message(
        self,
        text: str,
        embedding_client: EmbeddingClient,
        notes: str = "",
    ) -> Intervention:
        """Add message with intervention logging."""
        embedding = embedding_client.embed_single(text)
        if embedding is None:
            raise ValueError("Embedding generation failed")

        vector_id = self.add(
            text=text,
            embedding=embedding,
            mind_id=-1,
            extra_metadata={"injected": True},
        )

        intervention = self.intervention_log.record(
            intervention_type=INTERVENTION_INJECT,
            content={"text": text, "vector_id": vector_id},
            snapshot_before=f"{self.current_branch}@{self.iteration}",
            snapshot_after=f"{self.current_branch}@{self.iteration}",
            notes=notes,
        )

        self._save()
        return intervention

    def record_run(self, iterations_run: int, notes: str = "") -> Intervention:
        """Record a run intervention."""
        intervention = self.intervention_log.record(
            intervention_type=INTERVENTION_RUN,
            content={
                "iterations": iterations_run,
                "branch": self.current_branch,
                "end_iteration": self.iteration,
            },
            snapshot_before=f"{self.current_branch}@{self.iteration - iterations_run}",
            snapshot_after=f"{self.current_branch}@{self.iteration}",
            notes=notes,
        )
        self._save()
        return intervention

    def get_status(self) -> dict:
        """Get current session status."""
        visible_ids = self.get_visible_ids()
        return {
            "session_dir": str(self.session_dir),
            "current_branch": self.current_branch,
            "iteration": self.iteration,
            "total_messages": self.vector_db.size() if self.vector_db else 0,
            "visible_messages": len(visible_ids),
            "active_pool_size": min(len(visible_ids), self.active_pool_size),
            "branches": list(self.branches.keys()),
            "interventions": len(self.intervention_log.all()),
        }

    def list_branches(self) -> list[dict]:
        """List all branches with details."""
        result = []
        for name, branch in self.branches.items():
            # Count messages in branch
            own_ids = 0
            for start, end in branch.id_ranges:
                if end is None:
                    end = self.vector_db.size() - 1
                own_ids += max(0, end - start + 1)

            result.append({
                "name": name,
                "current": name == self.current_branch,
                "parent": branch.parent,
                "parent_iteration": branch.parent_iteration,
                "own_messages": own_ids,
            })
        return result
