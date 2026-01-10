"""
Novelty injection hook.

Injects sampled content from an external file at fixed intervals
to introduce novel perspectives into the pool.
"""

import random
from pathlib import Path

# Configuration - adjust these for your use case
CONTENT_FILE = Path(__file__).parent / "novelty_content.txt"
INJECT_EVERY = 10  # Inject every N iterations
SAMPLE_COUNT = 1   # Number of items to inject per interval


def load_content() -> list[str]:
    """Load content items from file (one per line, blank lines separate items)."""
    if not CONTENT_FILE.exists():
        raise FileNotFoundError(f"Content file not found: {CONTENT_FILE}")

    text = CONTENT_FILE.read_text()

    # Split on double newlines for multi-line items, or single newlines for simple lists
    if "\n\n" in text:
        items = [item.strip() for item in text.split("\n\n") if item.strip()]
    else:
        items = [line.strip() for line in text.splitlines() if line.strip()]

    # Filter out comment lines
    items = [item for item in items if not item.startswith("#")]

    return items


def hook(session, iteration: int, runner) -> None:
    """
    Called before each iteration.

    Args:
        session: The Session object
        iteration: Current iteration number
        runner: The LogosRunner instance (provides embedding_client)
    """
    if iteration % INJECT_EVERY != 0:
        return

    if iteration == 0:
        # Skip iteration 0 (initial state)
        return

    content_items = load_content()
    if not content_items:
        print(f"[novelty] Warning: No content items found in {CONTENT_FILE}")
        return

    # Sample items to inject
    sample_size = min(SAMPLE_COUNT, len(content_items))
    selected = random.sample(content_items, sample_size)

    for text in selected:
        session.inject_message(
            text=text,
            embedding_client=runner.embedding_client,
            notes=f"novelty hook @ iteration {iteration}",
        )
        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"[novelty] Injected: {preview}")
