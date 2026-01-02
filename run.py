#!/usr/bin/env python3
"""
Entry point for Logosphere experiments.

Usage:
    python run.py [--name NAME] [--rounds N]

Example:
    python run.py --name first-test --rounds 5
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import config
from pool import Pool
from logger import Logger
from orchestrator import Orchestrator
from init_parser import load_init_file


def print_header():
    """Print experiment header."""
    print("=" * 60)
    print("LOGOSPHERE")
    print("Memetic dynamics experiment")
    print("=" * 60)
    print()


def save_config_snapshot(exp_dir: Path) -> None:
    """Save experiment configuration snapshot."""
    config_data = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "N_MINDS": config.N_MINDS,
            "K_SAMPLES": config.K_SAMPLES,
            "M_ACTIVE_POOL": config.M_ACTIVE_POOL,
            "MAX_ROUNDS": config.MAX_ROUNDS,
            "TOKEN_LIMIT": config.TOKEN_LIMIT,
        },
        "api": {
            "MODEL": config.MODEL,
            "API_BASE_URL": config.API_BASE_URL,
        },
        "system_prompt": config.SYSTEM_PROMPT,
        "init_signature": config.INIT_SIGNATURE,
    }

    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"✓ Config snapshot: {config_path}")


def run_experiment(name: str = None, rounds: int = None) -> None:
    """
    Run Logosphere experiment.

    Args:
        name: Experiment name (default: timestamp)
        rounds: Number of rounds (default: config.MAX_ROUNDS)
    """
    print_header()

    # Create experiment directory
    exp_dir = config.create_experiment_dir(name)
    print(f"Experiment: {exp_dir.name}")
    print(f"Directory: {exp_dir}")
    print()

    # Check for init.md
    init_path = exp_dir / "init.md"
    if not init_path.exists():
        print(f"❌ Error: init.md not found")
        print()
        print(f"Expected: {init_path}")
        print()
        print("To create init.md:")
        print(f"  1. cp {config.INIT_TEMPLATE} {init_path}")
        print(f"  2. Edit {init_path} to customize seed messages")
        print(f"  3. Run experiment again")
        print()
        sys.exit(1)

    # Load init.md
    print(f"Loading init.md...")
    seed_messages, init_signature = load_init_file(init_path)
    print(f"✓ Loaded {len(seed_messages)} seed messages")
    print(f"✓ Init signature: '{init_signature}'")
    print()

    # Save config snapshot
    save_config_snapshot(exp_dir)
    print()

    # Initialize pool with seeds
    pool = Pool(max_active=config.M_ACTIVE_POOL)
    for msg in seed_messages:
        pool.add_message(msg)

    print(f"Pool initialized:")
    print(f"  Total messages: {pool.size()}")
    print(f"  Active pool: {pool.active_size()} (tail {config.M_ACTIVE_POOL})")
    print()

    # Determine rounds
    max_rounds = rounds if rounds is not None else config.MAX_ROUNDS

    # Initialize logger
    log_dir = exp_dir / "logs"
    with Logger(log_dir) as logger:
        print(f"Logging to: {log_dir / 'experiment.jsonl'}")
        print()

        # Log experiment start
        logger.log_experiment_start(
            config={
                "N_MINDS": config.N_MINDS,
                "K_SAMPLES": config.K_SAMPLES,
                "M_ACTIVE_POOL": config.M_ACTIVE_POOL,
                "MAX_ROUNDS": max_rounds,
                "TOKEN_LIMIT": config.TOKEN_LIMIT,
                "MODEL": config.MODEL,
            },
            init_signature=init_signature,
            num_seeds=len(seed_messages)
        )

        # Create orchestrator
        orchestrator = Orchestrator(pool, logger)

        # Run experiment
        print(f"Running {max_rounds} rounds with {config.N_MINDS} Minds...")
        print()

        try:
            for round_num in range(1, max_rounds + 1):
                print(f"Round {round_num}/{max_rounds}...", end=" ", flush=True)
                messages_added = orchestrator.run_round(round_num)
                print(f"✓ ({messages_added} messages, {pool.size()} total)")

            print()
            print("=" * 60)
            print("EXPERIMENT COMPLETE")
            print("=" * 60)
            print()
            print(f"Rounds completed: {max_rounds}")
            print(f"Final pool size: {pool.size()}")
            print(f"Total tokens: {orchestrator.total_tokens:,}")
            print()

            # Log experiment end
            logger.log_experiment_end(
                total_rounds=max_rounds,
                final_pool_size=pool.size(),
                total_tokens=orchestrator.total_tokens
            )

            print(f"Logs: {log_dir / 'experiment.jsonl'}")
            print()

        except KeyboardInterrupt:
            print()
            print()
            print("Experiment interrupted by user")
            print(f"Completed rounds: {round_num - 1}/{max_rounds}")
            print(f"Pool size: {pool.size()}")
            print(f"Tokens used: {orchestrator.total_tokens:,}")
            print()
            sys.exit(1)


def main():
    """Parse arguments and run experiment."""
    parser = argparse.ArgumentParser(
        description="Run Logosphere memetic dynamics experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py
  python run.py --name first-test
  python run.py --name quick-test --rounds 5
  python run.py --rounds 20

Notes:
  - Experiment directory: experiments/<name>/
  - Must create init.md in experiment directory before running
  - Use 'cp init-template.txt experiments/<name>/init.md' to start
        """
    )

    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name (default: timestamp)'
    )

    parser.add_argument(
        '--rounds',
        type=int,
        default=None,
        help=f'Number of rounds (default: {config.MAX_ROUNDS})'
    )

    args = parser.parse_args()

    run_experiment(name=args.name, rounds=args.rounds)


if __name__ == "__main__":
    main()
