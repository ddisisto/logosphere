"""
Test orchestrator.py functionality

Note: This doesn't make real API calls - we'll test that end-to-end in run.py
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from src.core.orchestrator import Orchestrator
from src.core.pool import Pool
from src.core.logger import Logger
from src import config


def test_orchestrator_round():
    """Test orchestrator round execution with mocked Mind invocations."""
    print("Testing Orchestrator.run_round with mocked Minds...")

    # Create temp directory for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"

        # Initialize components
        pool = Pool(max_active=config.M_ACTIVE_POOL)
        logger = Logger(log_dir)

        # Add some seed messages
        for i in range(5):
            pool.add_message(f"seed message {i}")

        orchestrator = Orchestrator(pool, logger)

        # Mock invoke_mind to avoid API calls
        mock_result = {
            'thinking': 'Mock thinking',
            'transmitted': ['mock response'],
            'tokens_used': 100,
            'raw_output': 'Mock thinking\n---\nmock response'
        }

        with patch('orchestrator.invoke_mind', return_value=mock_result):
            # Run one round
            messages_added = orchestrator.run_round(round_num=1)

            print(f"  ✓ Round completed")
            print(f"  ✓ Messages added: {messages_added}")
            assert messages_added == config.N_MINDS, \
                f"Expected {config.N_MINDS} messages (one per Mind), got {messages_added}"

            # Check pool grew
            expected_size = 5 + config.N_MINDS
            assert pool.size() == expected_size, \
                f"Pool should have {expected_size} messages, has {pool.size()}"
            print(f"  ✓ Pool size: {pool.size()} (5 seeds + {config.N_MINDS} new)")

            # Check tokens tracked
            assert orchestrator.total_tokens == 100 * config.N_MINDS, \
                f"Expected {100 * config.N_MINDS} tokens, got {orchestrator.total_tokens}"
            print(f"  ✓ Tokens tracked: {orchestrator.total_tokens}")

        logger.close()


def test_orchestrator_multiple_rounds():
    """Test orchestrator running multiple rounds."""
    print("\nTesting Orchestrator.run with multiple rounds...")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"

        pool = Pool(max_active=config.M_ACTIVE_POOL)
        logger = Logger(log_dir)

        # Seed pool
        pool.add_message("initial seed")

        orchestrator = Orchestrator(pool, logger)

        # Mock Mind that produces one message per invocation
        mock_result = {
            'thinking': 'thinking',
            'transmitted': ['response'],
            'tokens_used': 50,
            'raw_output': 'thinking\n---\nresponse'
        }

        with patch('orchestrator.invoke_mind', return_value=mock_result):
            # Run 3 rounds
            orchestrator.run(max_rounds=3)

            print(f"  ✓ Completed 3 rounds")

            # After 3 rounds: 1 seed + (3 rounds × 3 minds × 1 message) = 10 messages
            expected_total = 1 + (3 * config.N_MINDS * 1)
            assert pool.size() == expected_total, \
                f"Expected {expected_total} messages, got {pool.size()}"
            print(f"  ✓ Pool size: {pool.size()}")

            # Tokens: 3 rounds × 3 minds × 50 tokens = 450
            expected_tokens = 3 * config.N_MINDS * 50
            assert orchestrator.total_tokens == expected_tokens
            print(f"  ✓ Total tokens: {orchestrator.total_tokens}")

        logger.close()


def test_orchestrator_all_messages_accepted():
    """Test that orchestrator accepts all transmitted messages."""
    print("\nTesting all messages accepted...")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        pool = Pool(max_active=config.M_ACTIVE_POOL)
        logger = Logger(log_dir)

        orchestrator = Orchestrator(pool, logger)

        # Mock Mind that outputs various messages
        mock_result = {
            'thinking': 'various thoughts',
            'transmitted': ['message 1', 'message 2', 'blank:', ''],
            'tokens_used': 50,
            'raw_output': 'thinking\n---\nmessage 1\n---\nmessage 2\n---\nblank:\n---\n'
        }

        with patch('orchestrator.invoke_mind', return_value=mock_result):
            orchestrator.run_round(round_num=1)

            # All transmitted messages should be added (4 messages × 3 minds = 12)
            expected = 4 * config.N_MINDS
            assert pool.size() == expected
            print(f"  ✓ All messages accepted (including blank messages)")
            print(f"  ✓ No filtering applied")

        logger.close()


def run_all_tests():
    """Run all orchestrator tests."""
    print("=" * 60)
    print("ORCHESTRATOR VALIDATION")
    print("=" * 60)
    print()

    test_orchestrator_round()
    test_orchestrator_multiple_rounds()
    test_orchestrator_all_messages_accepted()

    print()
    print("=" * 60)
    print("✅ ALL ORCHESTRATOR TESTS PASSED")
    print("=" * 60)
    print()
    print("Component validated:")
    print("  - Round execution works")
    print("  - Multiple rounds work")
    print("  - Pool deltas tracked correctly")
    print("  - Token accounting works")
    print("  - All messages accepted (no filtering)")


if __name__ == "__main__":
    run_all_tests()
