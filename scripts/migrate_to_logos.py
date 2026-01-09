#!/usr/bin/env python3
"""
One-off migration script: old vector_db format -> logos format.

Rules:
- Add branch: "main" to all messages
- Messages with '>>> ' prefix get injected: true (except vector_id 49)
- Messages before vector_id 524 without sampled_ids get sampled_ids: []
- Reorder fields: branch, round, mind_id, vector_id, timestamp, extras, text last
- Parsing artifacts ("```") left as-is
"""

import json
import shutil
from pathlib import Path

import numpy as np


def migrate(src_dir: Path, dst_dir: Path):
    """Migrate old vector_db to logos format."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # Create destination
    dst_dir.mkdir(parents=True, exist_ok=True)
    vector_db_dir = dst_dir / "vector_db"
    vector_db_dir.mkdir(exist_ok=True)

    # Copy embeddings as-is
    src_embeddings = src_dir / "embeddings.npy"
    dst_embeddings = vector_db_dir / "embeddings.npy"
    if src_embeddings.exists():
        shutil.copy(src_embeddings, dst_embeddings)
        print(f"Copied embeddings: {src_embeddings} -> {dst_embeddings}")

    # Process metadata
    src_metadata = src_dir / "metadata.jsonl"
    dst_metadata = vector_db_dir / "metadata.jsonl"

    migrated = 0
    injected_count = 0
    sampled_ids_added = 0

    with open(src_metadata) as f_in, open(dst_metadata, 'w') as f_out:
        for line in f_in:
            old = json.loads(line.strip())

            vector_id = old['vector_id']
            text = old['text']

            # Build new record with proper field order
            new = {
                'branch': 'main',
                'round': old['round'],
                'mind_id': old['mind_id'],
                'vector_id': vector_id,
                'timestamp': old['timestamp'],
            }

            # Check for injection (>>> prefix, except vector_id 49)
            is_injection = text.startswith('>>> ') and vector_id != 49
            if is_injection:
                new['injected'] = True
                injected_count += 1

            # Add sampled_ids if missing (before vector_id 524)
            if 'sampled_ids' in old:
                new['sampled_ids'] = old['sampled_ids']
            elif vector_id < 524:
                new['sampled_ids'] = []
                sampled_ids_added += 1

            # Copy any other extra fields
            for key in old:
                if key not in ('vector_id', 'text', 'round', 'mind_id', 'timestamp', 'sampled_ids'):
                    new[key] = old[key]

            # Text always last
            new['text'] = text

            f_out.write(json.dumps(new) + '\n')
            migrated += 1

    print(f"Migrated {migrated} messages")
    print(f"  Marked as injected: {injected_count}")
    print(f"  Added empty sampled_ids: {sampled_ids_added}")

    # Create branches.json
    # Determine max round for iteration count
    max_round = 0
    with open(dst_metadata) as f:
        for line in f:
            msg = json.loads(line)
            max_round = max(max_round, msg.get('round', 0))

    branches_data = {
        "current": "main",
        "iteration": max_round + 1,  # Next iteration after last
        "config": {},  # Will be set on first run
        "branches": {
            "main": {
                "name": "main",
                "parent": None,
                "parent_iteration": None
            }
        }
    }

    with open(dst_dir / "branches.json", 'w') as f:
        json.dump(branches_data, f, indent=2)

    print(f"Created branches.json (iteration: {max_round + 1})")
    print(f"\nMigration complete: {dst_dir}")
    print(f"Open with: python scripts/logos.py open {dst_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python scripts/migrate_to_logos.py <src_vector_db> <dst_session>")
        print("Example: python scripts/migrate_to_logos.py .daniel/out/vector_db .daniel/logos-session")
        sys.exit(1)

    migrate(Path(sys.argv[1]), Path(sys.argv[2]))
