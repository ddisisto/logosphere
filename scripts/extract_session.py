#!/usr/bin/env python3
"""
Extract messages from a logos session branch into a new session.

Reads from old format (branches.json, branch field in metadata).
Writes to new simplified format (session.json, no branch field).

Copies a range of visible messages (including parent branch history) into a
fresh session, renumbering vector IDs while preserving round numbers and metadata.

Usage:
    python scripts/extract_session.py --to-session ./new-session
    python scripts/extract_session.py --from-vid 10 --to-vid 50 --to-session ./new-session
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.core.session import Session

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
) -> tuple[int, int, int]:
    """
    Extract messages from source session branch into a new session.

    Reads old format (with branches), writes new simplified format.

    Args:
        source_session: Source session (current branch will be used)
        to_session_path: Path for new session directory
        from_vid: First vector_id to include (None = first visible)
        to_vid: Last vector_id to include (None = last in current branch)

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

    # Check destination doesn't exist (check both old and new format)
    if to_session_path.exists():
        if (to_session_path / "branches.json").exists():
            raise ValueError(f"Session (old format) already exists at {to_session_path}")
        if (to_session_path / "session.json").exists():
            raise ValueError(f"Session already exists at {to_session_path}")

    # Get source config
    source_config = source_session.config.copy()

    # Determine embedding dim from source
    embedding_dim = source_session.vector_db.embedding_dim

    # Create new session directory and vector_db subdirectory
    to_session_path.mkdir(parents=True, exist_ok=True)
    vector_db_path = to_session_path / "vector_db"
    vector_db_path.mkdir(parents=True, exist_ok=True)

    # Collect embeddings and metadata for new session
    new_embeddings = []
    new_metadata = []

    for new_vid, old_vid in enumerate(extract_ids):
        old_meta = source_session.vector_db.get_message(old_vid)
        old_embedding = source_session.vector_db.embeddings[old_vid]

        # Build new metadata: preserve all fields except vector_id and branch
        # Field order: round, mind_id, vector_id, timestamp, [extras], text
        new_meta = {
            'round': old_meta.get('round', 0),
            'mind_id': old_meta.get('mind_id', 0),
            'vector_id': new_vid,
            'timestamp': old_meta.get('timestamp', datetime.now(timezone.utc).isoformat()),
        }

        # Copy extra fields (injected, sampled_ids, seed, etc.) but not branch
        for key, value in old_meta.items():
            if key in ('round', 'mind_id', 'vector_id', 'timestamp', 'branch', 'text'):
                continue
            new_meta[key] = value

        # Text always last for readability
        new_meta['text'] = old_meta['text']

        new_embeddings.append(np.array(old_embedding).astype(np.float32))
        new_metadata.append(new_meta)

    # Save embeddings as numpy array
    if new_embeddings:
        np.save(vector_db_path / 'embeddings.npy', np.array(new_embeddings))

    # Save metadata as JSONL
    with open(vector_db_path / 'metadata.jsonl', 'w') as f:
        for meta in new_metadata:
            f.write(json.dumps(meta) + '\n')

    # Determine final iteration (max round in extracted messages)
    max_round = max(meta['round'] for meta in new_metadata)

    # Create session.json (new simplified format - no branches)
    session_data = {
        "iteration": max_round,
        "config": source_config,
    }
    with open(to_session_path / "session.json", 'w') as f:
        json.dump(session_data, f, indent=2)

    # Create empty intervention log
    (to_session_path / "interventions.jsonl").touch()

    return len(extract_ids), from_vid, to_vid


def main():
    parser = argparse.ArgumentParser(
        description="Extract messages from a logos session branch into a new session (new format)"
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

    args = parser.parse_args()

    # Load source session (old format with branches)
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
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"\nExtracted {count} messages (vid {from_vid} to {to_vid})")
    print(f"New session (simplified format): {args.to_session}")
    print(f"\nNote: New session uses session.json format (no branches)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
