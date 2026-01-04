#!/usr/bin/env python3
"""
Entry point for Logosphere experiments.

Usage:
    python run.py --name <experiment-name>

Workflow:
    1. mkdir experiments/<name>
    2. cp config-template.json experiments/<name>/config.json
    3. cp init-template.txt experiments/<name>/init.md
    4. Edit config.json and init.md
    5. python run.py --name <name>

Each experiment is self-contained with its own config.json and init.md.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.core.pool import Pool
from src.core.logger import Logger
from src.core.orchestrator import Orchestrator
from src.core.init_parser import load_init_file


def print_header():
    """Print experiment header."""
    print("=" * 60)
    print("LOGOSPHERE")
    print("Memetic dynamics experiment")
    print("=" * 60)
    print()


def load_experiment_config(exp_dir: Path) -> dict:
    """
    Load experiment configuration from config.json.

    Returns dict with all config values needed for experiment.
    """
    config_path = exp_dir / "config.json"

    if not config_path.exists():
        print(f"❌ Error: config.json not found")
        print()
        print(f"Expected: {config_path}")
        print()
        print("To create config.json:")
        print(f"  cp config-template.json {config_path}")
        print(f"  # Edit {config_path} to customize parameters")
        print()
        sys.exit(1)

    with open(config_path) as f:
        return json.load(f)


def create_from_template(name: str, template_name: str, model: str = None) -> None:
    """
    Create new experiment from template experiment.

    Copies config.json and init.md from template to new experiment directory.

    Args:
        name: New experiment name
        template_name: Existing experiment to use as template
        model: Optional model name to override in config.json

    # TODO: Track genealogy - record template source in config metadata
    # This will allow tracking experiment lineage and parameter evolution
    """
    template_dir = config.EXPERIMENTS_DIR / template_name
    new_dir = config.EXPERIMENTS_DIR / name

    # Validate template exists
    if not template_dir.exists():
        print(f"❌ Error: Template experiment not found")
        print(f"Expected: {template_dir}")
        print()
        sys.exit(1)

    # Check if target already exists
    if new_dir.exists():
        print(f"❌ Error: Experiment directory already exists")
        print(f"Location: {new_dir}")
        print()
        print("Choose a different name or remove the existing directory")
        print()
        sys.exit(1)

    # Create new directory
    new_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created: {new_dir}")

    # Copy config.json
    template_config = template_dir / "config.json"
    new_config = new_dir / "config.json"
    if template_config.exists():
        import shutil
        shutil.copy2(template_config, new_config)
        print(f"Copied: config.json from {template_name}")
    else:
        print(f"⚠ Warning: Template has no config.json")

    # Copy init.md
    template_init = template_dir / "init.md"
    new_init = new_dir / "init.md"
    if template_init.exists():
        import shutil
        shutil.copy2(template_init, new_init)
        print(f"Copied: init.md from {template_name}")
    else:
        print(f"⚠ Warning: Template has no init.md")

    # Override model if specified
    if model:
        if new_config.exists():
            with open(new_config, 'r') as f:
                cfg = json.load(f)
            cfg['api']['MODEL'] = model
            with open(new_config, 'w') as f:
                json.dump(cfg, f, indent=2)
            print(f"Updated: MODEL={model}")
        else:
            print(f"⚠ Warning: Cannot override MODEL - no config.json")

    print()
    print(f"✓ Experiment '{name}' created from template '{template_name}'")
    if model:
        print(f"  Model: {model}")
        print("  Running experiment...")
    print()


def run_experiment(name: str, analyze: bool = True, model: str = None) -> None:
    """
    Run Logosphere experiment.

    Args:
        name: Experiment name (REQUIRED - must be an existing directory)
        analyze: Run analysis after experiment (default: True)
        model: Optional model name to override in config.json before running

    Workflow:
        1. Create experiments/<name>/ directory
        2. Copy config-template.json to experiments/<name>/config.json and edit
        3. Copy init-template.txt to experiments/<name>/init.md and edit
        4. Run: python run.py <name>
    """
    print_header()

    # Verify experiment directory exists
    exp_dir = config.EXPERIMENTS_DIR / name
    if not exp_dir.exists():
        print(f"❌ Error: Experiment directory not found")
        print()
        print(f"Expected: {exp_dir}")
        print()
        print("To create experiment:")
        print(f"  mkdir -p {exp_dir}")
        print(f"  cp config-template.json {exp_dir}/config.json")
        print(f"  cp init-template.txt {exp_dir}/init.md")
        print(f"  # Edit config.json and init.md")
        print(f"  python run.py {name}")
        print()
        sys.exit(1)

    print(f"Experiment: {name}")
    print(f"Directory: {exp_dir}")
    print()

    # Override model if specified
    if model:
        config_path = exp_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            cfg['api']['MODEL'] = model
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            print(f"Updated MODEL={model}")
            print()

    # Load experiment config
    print("Loading config.json...")
    exp_config = load_experiment_config(exp_dir)
    params = exp_config.get('parameters', {})
    api = exp_config.get('api', {})

    # Extract parameters
    N_MINDS = params.get('N_MINDS', config.N_MINDS)
    K_SAMPLES = params.get('K_SAMPLES', config.K_SAMPLES)
    M_ACTIVE_POOL = params.get('M_ACTIVE_POOL', config.M_ACTIVE_POOL)
    MAX_ROUNDS = params.get('MAX_ROUNDS', config.MAX_ROUNDS)
    TOKEN_LIMIT = params.get('TOKEN_LIMIT', config.TOKEN_LIMIT)
    MODEL = api.get('MODEL', config.MODEL)
    SYSTEM_PROMPT = exp_config.get('system_prompt', config.SYSTEM_PROMPT)

    print(f"✓ N_MINDS={N_MINDS}, K_SAMPLES={K_SAMPLES}, M_ACTIVE_POOL={M_ACTIVE_POOL}")
    print(f"✓ MAX_ROUNDS={MAX_ROUNDS}, TOKEN_LIMIT={TOKEN_LIMIT}")
    print()

    # Check for init.md
    init_path = exp_dir / "init.md"
    if not init_path.exists():
        print(f"❌ Error: init.md not found")
        print()
        print(f"Expected: {init_path}")
        print()
        print("To create init.md:")
        print(f"  cp {config.INIT_TEMPLATE} {init_path}")
        print(f"  # Edit {init_path} to customize seed messages")
        print()
        sys.exit(1)

    # Load init.md
    print(f"Loading init.md...")
    seed_messages = load_init_file(init_path)
    print(f"✓ Loaded {len(seed_messages)} seed messages")
    print()

    # Initialize pool with seeds
    pool = Pool(max_active=M_ACTIVE_POOL)
    for msg in seed_messages:
        pool.add_message(msg)

    print(f"Pool initialized:")
    print(f"  Total messages: {pool.size()}")
    print(f"  Active pool: {pool.active_size()} (tail {M_ACTIVE_POOL})")
    print()

    # Initialize logger
    log_dir = exp_dir / "logs"
    with Logger(log_dir) as logger:
        print(f"Logging to: {log_dir / 'experiment.jsonl'}")
        print()

        # Log experiment start
        logger.log_experiment_start(
            config={
                "N_MINDS": N_MINDS,
                "K_SAMPLES": K_SAMPLES,
                "M_ACTIVE_POOL": M_ACTIVE_POOL,
                "MAX_ROUNDS": MAX_ROUNDS,
                "TOKEN_LIMIT": TOKEN_LIMIT,
                "MODEL": MODEL,
            },
            init_signature="",  # No longer used
            num_seeds=len(seed_messages)
        )

        # Create orchestrator with experiment config
        orchestrator = Orchestrator(
            pool=pool,
            logger=logger,
            n_minds=N_MINDS,
            k_samples=K_SAMPLES,
            system_prompt=SYSTEM_PROMPT,
            token_limit=TOKEN_LIMIT
        )

        # Track novel memes (messages added during experiment)
        novel_memes = []

        # Run experiment
        print(f"Running {MAX_ROUNDS} rounds with {N_MINDS} Minds...")
        print()

        try:
            for round_num in range(1, MAX_ROUNDS + 1):
                print(f"Round {round_num}/{MAX_ROUNDS}...", end=" ", flush=True)

                # Run round
                messages_added = orchestrator.run_round(round_num)

                print(f"✓ ({messages_added} messages, {pool.size()} total)")

            print()
            print("=" * 60)
            print("EXPERIMENT COMPLETE")
            print("=" * 60)
            print()
            print(f"Rounds completed: {MAX_ROUNDS}")
            print(f"Final pool size: {pool.size()}")
            print(f"Total tokens: {orchestrator.total_tokens:,}")
            print()

            # Log experiment end
            logger.log_experiment_end(
                total_rounds=MAX_ROUNDS,
                final_pool_size=pool.size(),
                total_tokens=orchestrator.total_tokens
            )

            print(f"Logs: {log_dir / 'experiment.jsonl'}")
            print()

            # Run analysis if requested
            if analyze:
                print("Running analysis...")
                print()

                # Import and run novel_memes analysis
                import analyze as analyzer
                try:
                    analyzer.analyze_novel_memes(exp_dir)
                except Exception as e:
                    print(f"Warning: Analysis failed: {e}")
                    print(f"You can re-run analysis with: python analyze.py {name}")
                    print()
            else:
                print("Skipped analysis (use --analyze to enable)")
                print(f"Run analysis later with: python analyze.py {name}")
                print()

        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("EXPERIMENT INTERRUPTED")
            print("=" * 60)
            print()
            print(f"Completed {round_num - 1}/{MAX_ROUNDS} rounds")
            print(f"Current pool size: {pool.size()}")
            print(f"Tokens used: {orchestrator.total_tokens:,}")
            print()
            print(f"Logs: {log_dir / 'experiment.jsonl'}")
            print()
            print("To analyze partial results:")
            print(f"  python analyze.py {name}")
            print()
            sys.exit(1)


def main():
    """Parse arguments and run experiment."""
    parser = argparse.ArgumentParser(
        description="Run Logosphere memetic dynamics experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  # Quick model comparison (create from template + override model + run)
  python run.py sonnet-run --template _baseline --model anthropic/claude-sonnet-4.5
  python run.py haiku-run --template _baseline --model anthropic/claude-haiku-4.5

  # Create from template for manual editing
  python run.py new-experiment --template existing-experiment
  # Edit config.json and init.md as needed
  python run.py new-experiment

  # Override model in existing experiment
  python run.py my-experiment --model different-model/name

  # Create from scratch
  mkdir experiments/my-experiment
  cp config-template.json experiments/my-experiment/config.json
  cp init-template.txt experiments/my-experiment/init.md
  # Edit config.json and init.md
  python run.py my-experiment

Notes:
  - config.json: Set N_MINDS, MAX_ROUNDS, MODEL, etc.
  - init.md: Seed messages for the pool
  - Each experiment is self-contained in its directory
  - --model flag enables quick model comparisons with same config/seeds
        """
    )

    parser.add_argument(
        'name',
        type=str,
        help='Experiment name (directory under experiments/)'
    )

    parser.add_argument(
        '--analyze',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Run analysis after experiment (default: True)'
    )

    parser.add_argument(
        '--template',
        type=str,
        help='Create new experiment from existing experiment (template name)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Override MODEL in config.json (creates/updates then runs)'
    )

    args = parser.parse_args()

    # If template specified, create from template
    if args.template:
        create_from_template(name=args.name, template_name=args.template, model=args.model)
        # If model also specified, continue to run; otherwise exit
        if not args.model:
            print("Next steps:")
            print(f"  1. Edit experiments/{args.name}/config.json and init.md as needed")
            print(f"  2. Run: python run.py {args.name}")
            print()
            sys.exit(0)

    run_experiment(name=args.name, analyze=args.analyze, model=args.model)


if __name__ == "__main__":
    main()
