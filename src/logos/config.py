"""
Configuration for Logos reasoning sessions.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

# Re-export for type hints
__all__ = ["LogosConfig", "DEFAULT_SYSTEM_PROMPT", "EXTERNAL_PROMPT_PREFIX"]


# Default system prompt - minimal framing
DEFAULT_SYSTEM_PROMPT = """You receive thoughts from a shared pool.

Read them. Think privately. Transmit thoughts worth keeping.

Transmitted thoughts persist and compete for attention.
Thoughts not transmitted are forgotten.
Thoughts should be largely self-contained, but may reference or include others.
Thoughts may be retransmitted verbatim, and/or compressed, extended, decomposed, recomposed, etc.

Incoming prompts from external sources are prefixed with '>>> '.

Format: Thoughts (multi-line, unconstrained) separated by --- on its own line. e.g.

```
This is space for private thought and reasoning
---
This is the first transmitted thought
---
This is another
---
```
"""

# Prefix for externally-sourced prompts
EXTERNAL_PROMPT_PREFIX = ">>> "


@dataclass
class LogosConfig:
    """Configuration for a Logos reasoning session."""

    # Pool parameters
    k_samples: int = 5
    active_pool_size: int = 50

    # Termination (for batch mode) - None = disabled
    convergence_threshold: Optional[float] = None  # Coherence threshold (e.g., 0.75)
    convergence_coverage: Optional[float] = None   # Cluster coverage threshold (e.g., 0.50)
    stability_window: Optional[int] = None         # Iterations of stable clusters (e.g., 3)

    # Clustering
    min_cluster_size: int = 3

    # LLM
    model: str = "anthropic/claude-haiku-4.5"
    token_limit: int = 4000
    system_prompt: str = field(default_factory=lambda: DEFAULT_SYSTEM_PROMPT)

    # Embeddings
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536

    # Output
    verbose: bool = True

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LogosConfig":
        """Create from dict."""
        return cls(**data)
