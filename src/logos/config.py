"""
Configuration for Logos reasoning sessions.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

__all__ = [
    "LogosConfig",
    "DEFAULT_SYSTEM_PROMPT",
    "EXTERNAL_PROMPT_PREFIX",
    "load_api_key",
    "API_BASE_URL",
]

# External messages (injected/seeded) get this prefix
EXTERNAL_PROMPT_PREFIX = ">>> "


def load_api_key() -> str:
    """Load API key from environment or .env file."""
    # Try environment variable first
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API")
    if key:
        return key

    # Fall back to .env file
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API"):
                    # Handle both KEY:value and KEY=value formats
                    if ":" in line:
                        return line.split(":", 1)[1].strip()
                    elif "=" in line:
                        return line.split("=", 1)[1].strip()
    raise ValueError("API key not found. Set OPENROUTER_API_KEY env var or add to .env")


API_BASE_URL = "https://openrouter.ai/api/v1"


# Default system prompt - minimal framing
DEFAULT_SYSTEM_PROMPT = """You receive thoughts from a shared pool.

Read them. Think privately. Transmit thoughts worth keeping.

Transmitted thoughts persist and compete for attention.
Thoughts not transmitted are forgotten.
Thoughts should be largely self-contained, but may reference or include others.
Thoughts may be retransmitted verbatim, and/or compressed, extended, decomposed, recomposed, etc.

Incoming prompts from external sources are prefixed with '>>> '.

Format: Thoughts (multi-line, unconstrained) separated by --- on its own line. e.g.

This and all above is space for private thought and reasoning
---
This is the first transmitted thought
---
This is another
---
"""


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
    show_cluster_ids: bool = False  # Annotate sampled messages with [N], [~], [·]

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
        """Create from dict, filtering out unknown keys (e.g., hooks)."""
        # Get valid field names from dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
