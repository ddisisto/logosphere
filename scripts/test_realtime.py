#!/usr/bin/env python3
"""
Integration test for real-time detection infrastructure.

Runs a minimal 2-round experiment with:
- VectorDB (instead of Pool)
- Real-time embedding generation
- Attractor detection
- All new logging events

Usage:
    python scripts/test_realtime.py

Creates: experiments/_test_realtime/ (auto-cleaned on success)
"""

import sys
import time
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.vector_db import VectorDB
from src.core.embedding_client import EmbeddingClient, EmbeddingAPIError
from src.core.interventions import create_intervention
from src.core.logger import Logger
from src.analysis.attractors import AttractorDetector
from src.config import (
    EXPERIMENTS_DIR,
    DEFAULT_EMBEDDING_CONFIG,
    DEFAULT_ATTRACTOR_CONFIG,
    DEFAULT_INTERVENTION_CONFIG,
)


def run_integration_test():
    """Run minimal integration test."""
    print("=" * 60)
    print("Real-Time Detection Integration Test")
    print("=" * 60)

    # Setup
    exp_dir = EXPERIMENTS_DIR / "_test_realtime"
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True)
    (exp_dir / "logs").mkdir()

    try:
        # Initialize components
        print("\n1. Initializing components...")

        vector_db = VectorDB(active_pool_size=50, embedding_dim=1536)
        print(f"   VectorDB: active_pool_size=50")

        embedding_client = EmbeddingClient(enabled=True)
        print(f"   EmbeddingClient: model={embedding_client.model}")

        intervention = create_intervention({'strategy': 'none'}, vector_db)
        print(f"   Intervention: {type(intervention).__name__}")

        logger = Logger(exp_dir / "logs")
        print(f"   Logger: {logger.log_file}")

        # Log experiment start
        logger.log_experiment_start(
            config={
                'test': True,
                'embeddings': {'enabled': True},
                'attractor_detection': {'enabled': True},
            },
            init_signature="test",
            num_seeds=0,
        )

        # Simulate 2 rounds
        print("\n2. Running 2-round simulation...")

        test_messages = [
            # Round 1
            [
                "This is a test message about memetic dynamics.",
                "Attractors emerge from pool sampling patterns.",
                "The pool evolves through Mind iterations.",
            ],
            # Round 2
            [
                "Convergence is observed when messages become similar.",
                "Diversity can be maintained through intervention strategies.",
            ],
        ]

        all_vector_ids = []

        for round_num in range(2):
            print(f"\n   Round {round_num}:")
            msgs = test_messages[round_num]

            # Pre-round intervention hook
            intervention.on_round_start(round_num, None)

            # Simulate sampling (would come from intervention in real run)
            sample = intervention.on_sample(k=3, round_num=round_num)
            print(f"   - Sampled {len(sample)} messages from pool")

            logger.log_round_start(
                round_num=round_num,
                pool_size=vector_db.size(),
                active_pool_size=vector_db.active_size(),
            )

            # Embed new messages
            print(f"   - Embedding {len(msgs)} messages...")
            start_time = time.time()
            try:
                embeddings = embedding_client.embed_batch(msgs)
                latency_ms = (time.time() - start_time) * 1000
                print(f"   - Embeddings: shape={embeddings.shape}, latency={latency_ms:.0f}ms")

                logger.log_embedding_batch(
                    round_num=round_num,
                    num_texts=len(msgs),
                    latency_ms=latency_ms,
                    model=embedding_client.model,
                )
            except EmbeddingAPIError as e:
                print(f"   ❌ Embedding failed: {e}")
                logger.log_error(str(e), round_num=round_num, error_type="abort")
                raise

            # Add to VectorDB
            round_vector_ids = []
            for msg, emb in zip(msgs, embeddings):
                vid = vector_db.add(
                    text=msg,
                    embedding=emb,
                    round_num=round_num,
                    mind_id=0,
                )
                round_vector_ids.append(vid)
                all_vector_ids.append(vid)

            print(f"   - Added {len(round_vector_ids)} messages to VectorDB (IDs: {round_vector_ids})")

            # Detect attractors (need enough messages)
            if vector_db.active_size() >= 5:
                detector = AttractorDetector(vector_db, min_cluster_size=3)
                attractor_state = detector.detect(round_num)
                print(f"   - Attractors: {attractor_state['num_clusters']} clusters, {attractor_state['noise_count']} noise")

                logger.log_attractor_state(
                    round_num=round_num,
                    num_clusters=attractor_state['num_clusters'],
                    noise_count=attractor_state['noise_count'],
                    clusters=attractor_state['clusters'],
                    detected=attractor_state['detected'],
                )
            else:
                print(f"   - Attractors: skipped (need ≥5 messages)")

            # Post-round intervention hook
            intervention.on_round_end(round_num, None, round_vector_ids)

            # Log round end
            logger.log_round_end(
                round_num=round_num,
                messages_added=len(round_vector_ids),
                vector_ids=round_vector_ids,
            )

        # Save VectorDB
        print("\n3. Saving VectorDB...")
        db_path = exp_dir / "vector_db"
        vector_db.save(db_path)
        print(f"   Saved to: {db_path}")

        # Verify save/load
        print("\n4. Verifying save/load roundtrip...")
        loaded_db = VectorDB.load(db_path, active_pool_size=50)
        assert loaded_db.size() == vector_db.size(), "Size mismatch"
        print(f"   Loaded {loaded_db.size()} messages, active={loaded_db.active_size()}")

        # Log experiment end
        logger.log_experiment_end(
            total_rounds=2,
            final_pool_size=vector_db.size(),
            total_tokens=0,  # No actual LLM calls in this test
        )
        logger.close()

        # Verify log file
        print("\n5. Verifying log file...")
        import json
        with open(exp_dir / "logs" / "experiment.jsonl") as f:
            events = [json.loads(line) for line in f]

        event_types = [e['type'] for e in events]
        print(f"   Events: {event_types}")

        required_events = ['experiment_start', 'round_start', 'embedding_batch', 'round_end', 'experiment_end']
        for req in required_events:
            assert req in event_types, f"Missing event: {req}"
        print("   ✓ All required events present")

        # Success
        print("\n" + "=" * 60)
        print("✓ Integration test PASSED")
        print("=" * 60)

        # Cleanup on success
        shutil.rmtree(exp_dir)
        print(f"\nCleaned up: {exp_dir}")

        return True

    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        print(f"   Logs preserved at: {exp_dir}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
