#!/usr/bin/env python3
"""
Validation experiment for Phase D - VectorDB-based orchestrator.

Runs a 10-round experiment with full real-time detection stack:
- VectorDB storage (replaces Pool)
- Real-time embedding generation
- Attractor detection each round
- NoIntervention (baseline sampling)
- VectorDB persistence
- All logging events

Usage:
    python scripts/run_validation_experiment.py

Creates: experiments/_validation_phase_d/
"""

import sys
import time
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.vector_db import VectorDB
from src.core.logger import Logger
from src.core.orchestrator import Orchestrator, ExperimentAbortError
from src.core.embedding_client import EmbeddingClient, EmbeddingAPIError
from src.core.interventions import create_intervention
from src.analysis.attractors import AttractorDetector
from src.config import EXPERIMENTS_DIR


def run_validation():
    """Run Phase D validation experiment."""
    print("=" * 60)
    print("Phase D Validation Experiment")
    print("VectorDB-based Orchestrator with Real-Time Detection")
    print("=" * 60)
    print()

    # Setup
    exp_dir = EXPERIMENTS_DIR / "_validation_phase_d"
    if exp_dir.exists():
        print(f"Removing existing: {exp_dir}")
        shutil.rmtree(exp_dir)

    exp_dir.mkdir(parents=True)
    (exp_dir / "logs").mkdir()

    # Configuration
    N_MINDS = 1
    K_SAMPLES = 3
    M_ACTIVE_POOL = 50
    MAX_ROUNDS = 10
    TOKEN_LIMIT = 2000

    SYSTEM_PROMPT = """You receive messages from others. Read them.

Content before the first --- is private thinking, not transmitted.
After ---, write messages for others, separated by ---."""

    # Seed messages (simple, diverse)
    seed_messages = [
        "Ideas spread through networks of minds.",
        "Repetition is a form of emphasis.",
        "Novel thoughts compete for attention.",
        "Some patterns persist, others fade.",
        "The pool shapes what emerges next.",
    ]

    print(f"Config:")
    print(f"  N_MINDS={N_MINDS}, K_SAMPLES={K_SAMPLES}, M_ACTIVE_POOL={M_ACTIVE_POOL}")
    print(f"  MAX_ROUNDS={MAX_ROUNDS}, TOKEN_LIMIT={TOKEN_LIMIT}")
    print(f"  Seeds: {len(seed_messages)}")
    print()

    try:
        # Initialize VectorDB
        print("1. Initializing VectorDB...")
        vector_db = VectorDB(active_pool_size=M_ACTIVE_POOL)

        # Initialize embedding client (ENABLED for validation)
        print("2. Initializing EmbeddingClient...")
        embedding_client = EmbeddingClient(enabled=True)
        print(f"   Model: {embedding_client.model}")

        # Embed and add seed messages
        print("3. Embedding seed messages...")
        start = time.time()
        embeddings = embedding_client.embed_batch(seed_messages)
        latency = (time.time() - start) * 1000
        print(f"   Embedded {len(seed_messages)} seeds in {latency:.0f}ms")

        for msg, emb in zip(seed_messages, embeddings):
            vector_db.add(text=msg, embedding=emb, round_num=0, mind_id=-1)

        print(f"   VectorDB size: {vector_db.size()}, active: {vector_db.active_size()}")

        # Initialize intervention (NoIntervention)
        print("4. Initializing intervention...")
        intervention = create_intervention({'strategy': 'none'}, vector_db)
        print(f"   Type: {type(intervention).__name__}")

        # Initialize attractor detector
        print("5. Initializing AttractorDetector...")
        attractor_detector = AttractorDetector(
            vector_db=vector_db,
            min_cluster_size=3,  # Lower for validation with small pool
        )

        # Initialize logger
        print("6. Initializing Logger...")
        logger = Logger(exp_dir / "logs")

        # Log experiment start
        logger.log_experiment_start(
            config={
                "N_MINDS": N_MINDS,
                "K_SAMPLES": K_SAMPLES,
                "M_ACTIVE_POOL": M_ACTIVE_POOL,
                "MAX_ROUNDS": MAX_ROUNDS,
                "TOKEN_LIMIT": TOKEN_LIMIT,
                "embeddings": {"enabled": True},
                "attractor_detection": {"enabled": True, "min_cluster_size": 3},
                "interventions": {"strategy": "none"},
                "validation": True,
            },
            init_signature="phase_d_validation",
            num_seeds=len(seed_messages),
        )

        # Create orchestrator
        print("7. Creating Orchestrator...")
        orchestrator = Orchestrator(
            vector_db=vector_db,
            logger=logger,
            embedding_client=embedding_client,
            n_minds=N_MINDS,
            k_samples=K_SAMPLES,
            system_prompt=SYSTEM_PROMPT,
            token_limit=TOKEN_LIMIT,
            attractor_detector=attractor_detector,
            intervention=intervention,
            output_dir=exp_dir,
        )

        # Run experiment
        print()
        print(f"8. Running {MAX_ROUNDS} rounds...")
        print()

        for round_num in range(1, MAX_ROUNDS + 1):
            print(f"   Round {round_num}/{MAX_ROUNDS}...", end=" ", flush=True)
            start = time.time()

            messages_added = orchestrator.run_round(round_num)

            elapsed = time.time() - start
            print(f"✓ +{messages_added} msgs, {vector_db.size()} total ({elapsed:.1f}s)")

        print()

        # Log experiment end
        logger.log_experiment_end(
            total_rounds=MAX_ROUNDS,
            final_pool_size=vector_db.size(),
            total_tokens=orchestrator.total_tokens,
        )
        logger.close()

        # Save final VectorDB
        print("9. Saving VectorDB...")
        vector_db.save(exp_dir / "vector_db")

        # Verify save/load
        print("10. Verifying save/load roundtrip...")
        loaded_db = VectorDB.load(exp_dir / "vector_db", active_pool_size=M_ACTIVE_POOL)
        assert loaded_db.size() == vector_db.size(), f"Size mismatch: {loaded_db.size()} vs {vector_db.size()}"
        print(f"    ✓ Loaded {loaded_db.size()} messages")

        # Verify log file
        print("11. Verifying log events...")
        import json
        with open(exp_dir / "logs" / "experiment.jsonl") as f:
            events = [json.loads(line) for line in f]

        event_types = [e['type'] for e in events]
        event_counts = {t: event_types.count(t) for t in set(event_types)}
        print(f"    Events: {event_counts}")

        required_events = [
            'experiment_start',
            'round_start',
            'mind_invocation',
            'embedding_batch',
            'round_end',
            'experiment_end',
        ]
        for req in required_events:
            assert req in event_types, f"Missing event: {req}"
        print("    ✓ All required events present")

        # Check for attractor events (may not have any if not enough clustering)
        attractor_events = [e for e in events if e['type'] == 'attractor_state']
        print(f"    ✓ Attractor detection events: {len(attractor_events)}")

        # Summary
        print()
        print("=" * 60)
        print("✓ VALIDATION PASSED")
        print("=" * 60)
        print()
        print(f"Experiment: {exp_dir}")
        print(f"Rounds: {MAX_ROUNDS}")
        print(f"Final pool: {vector_db.size()} messages")
        print(f"Tokens: {orchestrator.total_tokens:,}")
        print()
        print("Files created:")
        print(f"  {exp_dir / 'logs' / 'experiment.jsonl'}")
        print(f"  {exp_dir / 'vector_db' / 'embeddings.npy'}")
        print(f"  {exp_dir / 'vector_db' / 'metadata.jsonl'}")
        print()

        return True

    except EmbeddingAPIError as e:
        print(f"\n❌ Embedding API Error: {e}")
        print("Check API key and network connection")
        return False

    except ExperimentAbortError as e:
        print(f"\n❌ Experiment Aborted: {e}")
        return False

    except Exception as e:
        print(f"\n❌ Validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
