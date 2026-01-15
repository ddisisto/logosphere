"""
Configuration for Mind v2 runner.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MindConfig:
    """Configuration for Mind v2 runner."""

    # Mind identity
    mind_id: str = "mind_0"

    # Sampling
    k_samples: int = 5  # Thoughts to sample from thinking pool

    # LLM
    model: str = "anthropic/claude-haiku-4.5"
    token_limit: int = 4000

    # Output
    verbose: bool = True
