"""
Parse init.md file to extract seed messages.

Reuses Mind parsing logic to extract messages from init file.
"""

from pathlib import Path
from mind import parse_output
from config import INIT_SIGNATURE


def load_init_file(init_path: Path) -> tuple[list[str], str]:
    """
    Load and parse init.md file to extract seed messages.

    Args:
        init_path: Path to init.md file

    Returns:
        (messages, signature) tuple where:
            messages: List of seed messages to add to pool
            signature: Extracted signature (should be INIT_SIGNATURE)

    Raises:
        ValueError: If init file not properly terminated or signature mismatch
    """
    if not init_path.exists():
        raise ValueError(f"Init file not found: {init_path}")

    # Read init file
    with open(init_path) as f:
        content = f.read()

    # Parse using Mind parsing logic
    thinking, messages, completed, signature = parse_output(content)

    # Validate proper termination
    if not completed:
        raise ValueError(
            f"Init file {init_path} not properly terminated. "
            "Must end with blank message (two --- with nothing between)."
        )

    # Validate signature matches expected
    if signature != INIT_SIGNATURE:
        raise ValueError(
            f"Init file signature mismatch. "
            f"Expected '{INIT_SIGNATURE}', got '{signature}'"
        )

    return messages, signature
