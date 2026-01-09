"""
Session management for Logosphere.

Branch-based model: single append-only VectorDB with branch field per message.
Visibility computed by filtering on branch name + parent lineage.
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

# External messages (injected/seeded) get this prefix
EXTERNAL_PROMPT_PREFIX = ">>> "


@dataclass
class Branch:
    """A branch is a view over the VectorDB."""

    name: str
    parent: Optional[str]  # Parent branch name
    parent_iteration: Optional[int]  # Iteration we branched at
    iteration: int = 0  # Current iteration for this branch
    parent_vector_id: Optional[int] = None  # If set, filter by vector_id instead of round
    config: dict = None  # Branch-specific config (inherited from parent on creation)

    def __post_init__(self):
        if self.config is None:
            self.config = {}

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "parent": self.parent,
            "parent_iteration": self.parent_iteration,
            "iteration": self.iteration,
            "config": self.config,
        }
        if self.parent_vector_id is not None:
            d["parent_vector_id"] = self.parent_vector_id
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Branch:
        # Handle legacy data without iteration field
        if 'iteration' not in data:
            data = data.copy()
            data['iteration'] = 0
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

        # Create main branch (iteration=0 is default in Branch)
        self.branches = {
            "main": Branch(
                name="main",
                parent=None,
                parent_iteration=None,
                iteration=0,
                config={},
            )
        }
        self.current_branch = "main"

        self._save()

    def _load(self) -> None:
        """Load session from disk."""
        # Load branches
        with open(self._branches_path) as f:
            data = json.load(f)

        self.current_branch = data["current"]
        self.branches = {
            name: Branch.from_dict(b)
            for name, b in data["branches"].items()
        }

        # Migrate legacy session-level iteration to current branch if needed
        if "iteration" in data and self.current.iteration == 0:
            self.current.iteration = data["iteration"]

        # Migrate session-level config to main branch if needed
        session_config = data.get("config", {})
        if session_config and not self.branches["main"].config:
            self.branches["main"].config = session_config

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

        # Save branches (iteration now stored per-branch)
        data = {
            "current": self.current_branch,
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

    @property
    def config(self) -> dict:
        """Current branch's config."""
        return self.current.config

    @config.setter
    def config(self, value: dict) -> None:
        """Set current branch's config."""
        self.current.config = value

    @property
    def iteration(self) -> int:
        """Current branch's iteration."""
        return self.current.iteration

    @iteration.setter
    def iteration(self, value: int) -> None:
        """Set current branch's iteration."""
        self.current.iteration = value

    def get_visible_ids(self) -> set[int]:
        """Get all vector_ids visible to current branch."""
        return self._get_branch_visible_ids(self.current_branch, None)

    def _get_branch_visible_ids(
        self,
        branch_name: str,
        up_to_round: Optional[int] = None,
        up_to_vector_id: Optional[int] = None,
    ) -> set[int]:
        """Get IDs visible to a branch, optionally filtered by round or vector_id."""
        visible = set()
        branch = self.branches.get(branch_name)
        if not branch:
            return visible

        # Scan metadata for messages belonging to this branch
        for vid in range(self.vector_db.size()):
            meta = self.vector_db.get_message(vid)
            if meta and meta.get('branch') == branch_name:
                # Apply filters
                if up_to_vector_id is not None:
                    if vid <= up_to_vector_id:
                        visible.add(vid)
                elif up_to_round is None or meta.get('round', 0) <= up_to_round:
                    visible.add(vid)

        # Add inherited IDs from parent (up to branch point)
        if branch.parent:
            # Use vector_id filter if set, otherwise use round filter
            if branch.parent_vector_id is not None:
                visible.update(self._get_branch_visible_ids(
                    branch.parent, up_to_vector_id=branch.parent_vector_id
                ))
            elif branch.parent_iteration is not None:
                effective_round = branch.parent_iteration
                if up_to_round is not None:
                    effective_round = min(effective_round, up_to_round)
                visible.update(self._get_branch_visible_ids(branch.parent, up_to_round=effective_round))

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
            branch=self.current_branch,
            extra_metadata=extra_metadata,
        )
        return vector_id

    def branch(self, name: str, from_vector_id: Optional[int] = None) -> str:
        """
        Create new branch from current state or historical point.

        Args:
            name: New branch name
            from_vector_id: If set, branch from this vector_id instead of current state

        Returns:
            Name of created branch
        """
        if name in self.branches:
            raise ValueError(f"Branch '{name}' already exists")

        # Determine branch point
        if from_vector_id is not None:
            # Branch from specific vector_id
            meta = self.vector_db.get_message(from_vector_id)
            if meta is None:
                raise ValueError(f"Vector ID {from_vector_id} not found")
            parent_iteration = meta.get('round', 0)
            parent_vector_id = from_vector_id
        else:
            # Branch from current state
            parent_iteration = self.iteration
            parent_vector_id = None

        # Create new branch (inherit parent's config and iteration)
        new_branch = Branch(
            name=name,
            parent=self.current_branch,
            parent_iteration=parent_iteration,
            iteration=parent_iteration,  # Start from parent's iteration at branch point
            parent_vector_id=parent_vector_id,
            config=self.config.copy(),  # Copy parent's config
        )
        self.branches[name] = new_branch

        # Log intervention
        content = {
            "from_branch": self.current_branch,
            "to_branch": name,
            "at_iteration": parent_iteration,
        }
        if parent_vector_id is not None:
            content["at_vector_id"] = parent_vector_id

        self.intervention_log.record(
            intervention_type=INTERVENTION_BRANCH,
            content=content,
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

        self.current_branch = name
        self._save()

    def inject_message(
        self,
        text: str,
        embedding_client: EmbeddingClient,
        notes: str = "",
    ) -> Intervention:
        """Add external message with intervention logging."""
        # Add external prompt prefix for consistency with seeded prompts
        prefixed_text = f"{EXTERNAL_PROMPT_PREFIX}{text}"

        embedding = embedding_client.embed_single(prefixed_text)
        if embedding is None:
            raise ValueError("Embedding generation failed")

        vector_id = self.add(
            text=prefixed_text,
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
        # Count messages per branch
        branch_counts = {name: 0 for name in self.branches}
        for vid in range(self.vector_db.size()):
            meta = self.vector_db.get_message(vid)
            if meta:
                branch_name = meta.get('branch')
                if branch_name in branch_counts:
                    branch_counts[branch_name] += 1

        result = []
        for name, branch in self.branches.items():
            result.append({
                "name": name,
                "current": name == self.current_branch,
                "parent": branch.parent,
                "parent_iteration": branch.parent_iteration,
                "own_messages": branch_counts[name],
            })
        return result
