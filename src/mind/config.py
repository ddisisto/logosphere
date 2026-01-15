"""
Configuration for Mind v2 runner.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MindConfig:
    """Runtime configuration for Mind v2 runner.

    Note: Session-level config (k_samples, model, etc.) is in SessionConfig.
    This is for runtime/invocation options only.
    """

    # Mind identity
    mind_id: str = "mind_0"

    # Output
    verbose: bool = True
