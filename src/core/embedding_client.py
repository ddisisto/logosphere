"""
Embedding client for generating vector embeddings via OpenRouter.

Provides batch embedding with abort-on-failure semantics for experiment integrity.
"""

import numpy as np
import requests
from typing import Optional

# Import directly from module to avoid circular import via logos.__init__
import os
from pathlib import Path

def _load_api_key() -> str:
    """Load API key from environment or .env file."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API")
    if key:
        return key
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API"):
                    if ":" in line:
                        return line.split(":", 1)[1].strip()
                    elif "=" in line:
                        return line.split("=", 1)[1].strip()
    raise ValueError("API key not found. Set OPENROUTER_API_KEY env var or add to .env")

API_BASE_URL = "https://openrouter.ai/api/v1"


class EmbeddingAPIError(Exception):
    """Raised when embedding API call fails. Triggers experiment abort."""
    pass


class EmbeddingClient:
    """
    Generate embeddings via OpenRouter API.

    Uses OpenAI's text-embedding-3-small model (1536 dimensions).
    Abort-on-failure: any API error raises EmbeddingAPIError.
    """

    # Default model - 1536 dimensions, good quality/cost tradeoff
    DEFAULT_MODEL = "openai/text-embedding-3-small"
    DEFAULT_DIM = 1536

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize embedding client.

        Args:
            model: Embedding model ID (default: openai/text-embedding-3-small)
            api_key: OpenRouter API key (default: from config)
            api_base_url: API base URL (default: from config)
            enabled: If False, embed_batch returns None (for testing without API)
        """
        self.model = model
        self.api_key = api_key or _load_api_key()
        self.api_base_url = api_base_url or API_BASE_URL
        self.enabled = enabled

        # Infer dimension from model
        if "text-embedding-3-small" in model:
            self.dim = 1536
        elif "text-embedding-3-large" in model:
            self.dim = 3072
        else:
            self.dim = self.DEFAULT_DIM

    def embed_batch(self, texts: list[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of strings to embed

        Returns:
            np.ndarray of shape (len(texts), dim) or None if disabled

        Raises:
            EmbeddingAPIError: On any API failure (triggers experiment abort)
        """
        if not self.enabled:
            return None

        if not texts:
            return np.array([]).reshape(0, self.dim)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            response = requests.post(
                f"{self.api_base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=30,  # 30s timeout for batch
            )
            response.raise_for_status()

            data = response.json()

            # Extract embeddings in order
            # API returns {data: [{embedding: [...], index: 0}, ...]}
            embeddings_data = sorted(data["data"], key=lambda x: x["index"])
            embeddings = [item["embedding"] for item in embeddings_data]

            return np.array(embeddings, dtype=np.float32)

        except requests.exceptions.Timeout:
            raise EmbeddingAPIError(f"Embedding API timeout after 30s for {len(texts)} texts")
        except requests.exceptions.RequestException as e:
            raise EmbeddingAPIError(f"Embedding API request failed: {e}")
        except (KeyError, IndexError, ValueError) as e:
            raise EmbeddingAPIError(f"Embedding API response parse error: {e}")

    def embed_single(self, text: str) -> Optional[np.ndarray]:
        """
        Convenience method for single text embedding.

        Args:
            text: String to embed

        Returns:
            np.ndarray of shape (dim,) or None if disabled

        Raises:
            EmbeddingAPIError: On any API failure
        """
        result = self.embed_batch([text])
        if result is None:
            return None
        return result[0]
