#!/usr/bin/env python3
"""
Entry point for Logosphere experiments.

Usage:
    python run.py [--name NAME]

Example:
    python run.py --name first-test

Notes:
    Edit config.py to customize experiment parameters (rounds, minds, etc.)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import yaml

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
    }

    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"✓ Config snapshot: {config_path}")


def save_novel_memes(exp_dir: Path, novel_memes: list[dict]) -> None:
    """
    Save novel memes (non-init messages) to YAML file.

    Args:
        exp_dir: Experiment directory
        novel_memes: List of dicts with keys: round, mind_id, content
    """
    novel_path = exp_dir / "novel_memes.yaml"

    with open(novel_path, 'w') as f:
        for i, meme in enumerate(novel_memes):
            if i > 0:
                f.write("---\n")

            # Write metadata
            f.write(f"round: {meme['round']}\n")
            f.write(f"mind_id: {meme['mind_id']}\n")

            # Write content as literal block scalar
            f.write("content: |-\n")
            for line in meme['content'].splitlines():
                f.write(f"  {line}\n")

    print(f"✓ Novel memes: {novel_path} ({len(novel_memes)} messages)")


def run_experiment(name: str = None) -> None:
    """
    Run Logosphere experiment.

    Args:
        name: Experiment name (default: timestamp)

    Notes:
        All parameters are read from config.py.
        Edit config.py before running to customize the experiment.
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
    seed_messages = load_init_file(init_path)
    print(f"✓ Loaded {len(seed_messages)} seed messages")
    print()

    # Initialize pool with seeds
    pool = Pool(max_active=config.M_ACTIVE_POOL)
    for msg in seed_messages:
        pool.add_message(msg)

    print(f"Pool initialized:")
    print(f"  Total messages: {pool.size()}")
    print(f"  Active pool: {pool.active_size()} (tail {config.M_ACTIVE_POOL})")
    print()

    # Use config.py values directly
    max_rounds = config.MAX_ROUNDS

    # Save config snapshot (read-only record of parameters used)
    save_config_snapshot(exp_dir)
    print()

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
                "MAX_ROUNDS": config.MAX_ROUNDS,
                "TOKEN_LIMIT": config.TOKEN_LIMIT,
                "MODEL": config.MODEL,
            },
            init_signature="",  # No longer used
            num_seeds=len(seed_messages)
        )

        # Create orchestrator
        orchestrator = Orchestrator(pool, logger)

        # Track novel memes (messages added during experiment)
        novel_memes = []

        # Run experiment
        print(f"Running {max_rounds} rounds with {config.N_MINDS} Minds...")
        print()

        try:
            for round_num in range(1, max_rounds + 1):
                print(f"Round {round_num}/{max_rounds}...", end=" ", flush=True)

                # Track pool size before round
                pool_size_before = pool.size()

                # Run round
                messages_added = orchestrator.run_round(round_num)

                # Extract novel memes added this round
                # Get the messages that were just added (pool delta)
                new_messages = pool.get_all()[pool_size_before:]

                # We need to get metadata from the log for these messages
                # For now, we'll track them from orchestrator's last round
                # The orchestrator logs the full context, so we can reconstruct

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

            # Extract novel memes from logs
            # Read experiment.jsonl and extract all mind_invocation events
            novel_memes = []
            log_path = log_dir / "experiment.jsonl"
            with open(log_path) as f:
                for line in f:
                    event = json.loads(line)
                    if event['type'] == 'mind_invocation':
                        # Add each transmitted message with metadata
                        for msg in event['transmitted']:
                            novel_memes.append({
                                'round': event['round'],
                                'mind_id': event['mind_id'],
                                'content': msg
                            })

            # Save novel memes to YAML
            save_novel_memes(exp_dir, novel_memes)
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

Notes:
  - Edit config.py to customize experiment parameters
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

    args = parser.parse_args()

    run_experiment(name=args.name)


if __name__ == "__main__":
    main()
