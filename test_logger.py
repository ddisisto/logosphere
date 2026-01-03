"""
Test logger.py functionality
"""

import json
import tempfile
from pathlib import Path
from logger import Logger


def test_logger():
    """Test logger writes JSONL correctly."""
    print("Testing Logger class...")

    # Use temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"

        # Create logger
        with Logger(log_dir) as logger:
            print(f"  ✓ Logger created in {log_dir}")

            # Log experiment start
            logger.log_experiment_start(
                config={"N_MINDS": 3, "K_SAMPLES": 3},
                init_signature="~init",
                num_seeds=17
            )
            print("  ✓ Logged experiment_start")

            # Log round start
            logger.log_round_start(
                round_num=1,
                pool_size=17,
                active_pool_size=15
            )
            print("  ✓ Logged round_start")

            # Log mind invocation
            logger.log_mind_invocation(
                round_num=1,
                mind_id=0,
                input_messages=["msg1", "msg2", "msg3"],
                thinking="I should respond carefully.",
                transmitted=["response message"],
                tokens_used=150,
                raw_output="I should respond carefully.\n---\nresponse message"
            )
            print("  ✓ Logged mind_invocation")

            # Log round end
            logger.log_round_end(
                round_num=1,
                messages_added=3,
                pool_delta=["new msg 1", "new msg 2", "new msg 3"]
            )
            print("  ✓ Logged round_end")

            # Log experiment end
            logger.log_experiment_end(
                total_rounds=10,
                final_pool_size=47,
                total_tokens=15000
            )
            print("  ✓ Logged experiment_end")

        # Verify file exists
        log_file = log_dir / "experiment.jsonl"
        assert log_file.exists(), "Log file should exist"
        print(f"  ✓ Log file created: {log_file}")

        # Read and verify events
        events = []
        with open(log_file) as f:
            for line in f:
                events.append(json.loads(line))

        assert len(events) == 5, f"Expected 5 events, got {len(events)}"
        print(f"  ✓ Read {len(events)} events from JSONL")

        # Verify event types
        expected_types = [
            "experiment_start",
            "round_start",
            "mind_invocation",
            "round_end",
            "experiment_end"
        ]
        actual_types = [e["type"] for e in events]
        assert actual_types == expected_types, f"Event types: {actual_types}"
        print(f"  ✓ Event types correct: {actual_types}")

        # Verify timestamps present
        for event in events:
            assert "timestamp" in event, "All events should have timestamp"
        print("  ✓ All events have timestamps")

        # Verify experiment_start data
        exp_start = events[0]
        assert exp_start["config"]["N_MINDS"] == 3
        assert exp_start["init_signature"] == "~init"
        assert exp_start["num_seed_messages"] == 17
        print("  ✓ experiment_start data correct")

        # Verify mind_invocation data
        mind_inv = events[2]
        assert mind_inv["round"] == 1
        assert mind_inv["mind_id"] == 0
        assert len(mind_inv["input_messages"]) == 3
        assert mind_inv["tokens_used"] == 150
        print("  ✓ mind_invocation data correct")

        # Verify round_end pool delta
        round_end = events[3]
        assert round_end["messages_added"] == 3
        assert len(round_end["pool_delta"]) == 3
        print("  ✓ round_end pool delta correct")


def run_all_tests():
    """Run all logger tests."""
    print("=" * 60)
    print("LOGGER VALIDATION")
    print("=" * 60)
    print()

    test_logger()

    print()
    print("=" * 60)
    print("✅ ALL LOGGER TESTS PASSED")
    print("=" * 60)
    print()
    print("Component validated:")
    print("  - JSONL writing correct")
    print("  - Typed events structure correct")
    print("  - All event types logged properly")
    print("  - Context manager works")


if __name__ == "__main__":
    run_all_tests()
