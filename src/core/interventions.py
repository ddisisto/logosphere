"""
Intervention infrastructure for dynamic sampling strategies.

Interventions are pluggable hooks that can modify sampling behavior
based on attractor state. Phase 1 implements only NoIntervention (baseline).

Future interventions (Phase 2+):
- Anti-convergence sampling (boost under-represented clusters)
- Diversity injection (sample from semantic periphery)
- Basin steering (guide pool toward specific attractors)
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .vector_db import VectorDB


class Intervention(ABC):
    """
    Abstract base class for sampling interventions.

    Interventions hook into the orchestrator at key points:
    - on_sample: Override how messages are sampled for Minds
    - on_round_start: React to attractor state before round
    - on_round_end: React after round completes

    Subclasses must implement on_sample. Other hooks are optional.
    """

    def __init__(self, vector_db: "VectorDB"):
        """
        Initialize intervention.

        Args:
            vector_db: VectorDB instance for sampling operations
        """
        self.vector_db = vector_db

    @abstractmethod
    def on_sample(self, k: int, round_num: int) -> list[str]:
        """
        Sample k messages for Mind input.

        This is the core intervention point. Override to implement
        custom sampling strategies.

        Args:
            k: Number of messages to sample
            round_num: Current round number

        Returns:
            List of k message texts
        """
        pass

    def on_round_start(self, round_num: int, attractor_state: Optional[dict]) -> None:
        """
        Called before each round starts.

        Override to react to attractor state, adjust internal state,
        or prepare for the round.

        Args:
            round_num: Current round number
            attractor_state: Last detection result (or None if not yet run)
        """
        pass

    def on_round_end(
        self,
        round_num: int,
        attractor_state: Optional[dict],
        vector_ids: list[int],
    ) -> None:
        """
        Called after each round ends.

        Override to update internal state based on what was produced.

        Args:
            round_num: Current round number
            attractor_state: Detection result after this round
            vector_ids: IDs of messages added this round
        """
        pass


class NoIntervention(Intervention):
    """
    Default intervention: standard uniform sampling from active pool.

    This is the baseline behavior matching the original Pool.sample().
    Used when no intervention strategy is configured.
    """

    def on_sample(self, k: int, round_num: int) -> list[str]:
        """Sample k messages uniformly at random from active pool."""
        return self.vector_db.sample_random(k, from_active_pool=True)


# Registry for intervention types
_INTERVENTION_REGISTRY = {
    "none": NoIntervention,
    "no_intervention": NoIntervention,
}


def register_intervention(name: str, cls: type) -> None:
    """
    Register an intervention class.

    Args:
        name: Name to register under
        cls: Intervention subclass
    """
    _INTERVENTION_REGISTRY[name] = cls


def create_intervention(
    config: dict,
    vector_db: "VectorDB",
) -> Intervention:
    """
    Create intervention from config.

    Args:
        config: Intervention config dict with 'strategy' key
        vector_db: VectorDB instance

    Returns:
        Configured Intervention instance

    Example config:
        {"strategy": "none"}  # NoIntervention
        {"strategy": "anti_convergence", "boost_factor": 2.0}  # Future
    """
    strategy = config.get("strategy", "none")

    if strategy not in _INTERVENTION_REGISTRY:
        raise ValueError(
            f"Unknown intervention strategy: {strategy}. "
            f"Available: {list(_INTERVENTION_REGISTRY.keys())}"
        )

    intervention_cls = _INTERVENTION_REGISTRY[strategy]
    return intervention_cls(vector_db=vector_db)
