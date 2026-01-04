"""
Parse init.md file to extract seed messages.

Reuses Mind parsing logic to extract messages from init file.
"""

from pathlib import Path
from .mind import parse_output


def load_init_file(init_path: Path) -> list[str]:
    """
    Load and parse init.md file to extract seed messages.

    Args:
        init_path: Path to init.md file

    Returns:
        List of seed messages to add to pool

    Raises:
        ValueError: If init file not found
    """
    if not init_path.exists():
        raise ValueError(f"Init file not found: {init_path}")

    # Read init file
    with open(init_path) as f:
        content = f.read()

    # Parse using Mind parsing logic
    thinking, messages = parse_output(content)

    return messages
