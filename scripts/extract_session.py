#!/usr/bin/env python3
"""
Extract messages from a logos session branch into a new session.

Copies a range of visible messages (including parent branch history) into a
fresh session, renumbering vector IDs while preserving round numbers and metadata.

Usage:
    python scripts/extract_session.py --to-session ./new-session
    python scripts/extract_session.py --from-vid 10 --to-vid 50 --to-session ./new-session
    python scripts/extract_session.py --to-session ./new-session --to-branch experiment
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.core.session import Session, Branch
from src.core.vector_db import VectorDB

# Same session tracking as logos.py
SESSION_FILE = Path.home() / ".logos_session"


def get_current_session_dir() -> Path:
    """Get the current session directory."""
    if SESSION_FILE.exists():
        return Path(SESSION_FILE.read_text().strip())
    raise RuntimeError("No session open. Use 'logos open' first.")


def extract_session(
    source_session: Session,
    to_session_path: Path,
    from_vid: int | None,
    to_vid: int | None,
    to_branch: str,
) -> tuple[int, int, int]:
    """
    Extract messages from source session branch into a new session.

    Args:
        source_session: Source session (current branch will be used)
        to_session_path: Path for new session directory
        from_vid: First vector_id to include (None = first visible)
        to_vid: Last vector_id to include (None = last in current branch)
        to_branch: Branch name in new session

    Returns:
        Tuple of (messages_extracted, first_vid, last_vid)
    """
    # Get all visible IDs for current branch
    visible_ids = source_session.get_visible_ids()
    if not visible_ids:
        raise ValueError("No visible messages in current branch")

    visible_sorted = sorted(visible_ids)

    # Determine range bounds
    if from_vid is None:
        from_vid = visible_sorted[0]
    if to_vid is None:
        # Last message in current branch (not parent)
        own_ids = [
            vid for vid in visible_sorted
            if source_session.vector_db.get_message(vid).get('branch') == source_session.current_branch
        ]
        if own_ids:
            to_vid = max(own_ids)
        else:
            # Branch has no own messages, use last visible
            to_vid = visible_sorted[-1]

    # Validate bounds
    if from_vid not in visible_ids:
        raise ValueError(f"from_vid {from_vid} is not visible to current branch")
    if to_vid not in visible_ids:
        raise ValueError(f"to_vid {to_vid} is not visible to current branch")
    if from_vid > to_vid:
        raise ValueError(f"from_vid ({from_vid}) must be <= to_vid ({to_vid})")

    # Filter to range [from_vid, to_vid] (inclusive)
    extract_ids = [vid for vid in visible_sorted if from_vid <= vid <= to_vid]

    if not extract_ids:
        raise ValueError("No messages in specified range")

    # Check destination doesn't exist
    if to_session_path.exists() and (to_session_path / "branches.json").exists():
        raise ValueError(f"Session already exists at {to_session_path}")

    # Get source config
    source_config = source_session.config.copy()

    # Determine embedding dim from source
    embedding_dim = source_session.vector_db.embedding_dim

    # Create new session directory
    to_session_path.mkdir(parents=True, exist_ok=True)

    # Create new VectorDB with extracted messages
    new_db = VectorDB(
        active_pool_size=source_config.get('active_pool_size', 50),
        embedding_dim=embedding_dim,
    )

    # Copy messages with renumbered vector_ids
    for new_vid, old_vid in enumerate(extract_ids):
        old_meta = source_session.vector_db.get_message(old_vid)
        old_embedding = source_session.vector_db.embeddings[old_vid]

        # Build new metadata preserving all fields except vector_id and branch
        new_meta = {}
        for key, value in old_meta.items():
            if key == 'vector_id':
                continue  # Will be set by add()
            elif key == 'branch':
                continue  # Will be set by add()
            elif key == 'text':
                continue  # Passed separately to add()
            else:
                new_meta[key] = value

        new_db.add(
            text=old_meta['text'],
            embedding=np.array(old_embedding),
            round_num=old_meta.get('round', 0),
            mind_id=old_meta.get('mind_id', 0),
            branch=to_branch,
            extra_metadata={k: v for k, v in new_meta.items() if k not in ('round', 'mind_id')},
        )

    # Save VectorDB
    vector_db_path = to_session_path / "vector_db"
    new_db.save(vector_db_path)

    # Determine final iteration (max round in extracted messages)
    max_round = max(
        source_session.vector_db.get_message(vid).get('round', 0)
        for vid in extract_ids
    )

    # Create branches.json with single branch
    import json
    branches_data = {
        "current": to_branch,
        "branches": {
            to_branch: {
                "name": to_branch,
                "parent": None,
                "parent_iteration": None,
                "iteration": max_round,
                "config": source_config,
            }
        }
    }
    with open(to_session_path / "branches.json", 'w') as f:
        json.dump(branches_data, f, indent=2)

    # Create empty intervention log
    (to_session_path / "interventions.jsonl").touch()

    return len(extract_ids), from_vid, to_vid


def main():
    parser = argparse.ArgumentParser(
        description="Extract messages from a logos session branch into a new session"
    )
    parser.add_argument(
        "--from-vid", type=int, default=None,
        help="First vector_id to include (default: first visible to branch)"
    )
    parser.add_argument(
        "--to-vid", type=int, default=None,
        help="Last vector_id to include (default: last in current branch)"
    )
    parser.add_argument(
        "--to-session", type=str, required=True,
        help="Path for new session directory"
    )
    parser.add_argument(
        "--to-branch", type=str, default="main",
        help="Branch name in new session (default: main)"
    )

    args = parser.parse_args()

    # Load source session
    try:
        source_dir = get_current_session_dir()
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    print(f"Source session: {source_dir}")
    source_session = Session(source_dir)
    print(f"Source branch: {source_session.current_branch}")

    visible_count = len(source_session.get_visible_ids())
    print(f"Visible messages: {visible_count}")

    # Extract
    try:
        count, from_vid, to_vid = extract_session(
            source_session=source_session,
            to_session_path=Path(args.to_session),
            from_vid=args.from_vid,
            to_vid=args.to_vid,
            to_branch=args.to_branch,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"\nExtracted {count} messages (vid {from_vid} to {to_vid})")
    print(f"New session: {args.to_session}")
    print(f"New branch: {args.to_branch}")
    print(f"\nTo open: logos open {args.to_session}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
