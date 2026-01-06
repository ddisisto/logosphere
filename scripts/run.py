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
from src.core.vector_db import VectorDB
from src.core.logger import Logger
from src.core.orchestrator import Orchestrator, ExperimentAbortError
from src.core.embedding_client import EmbeddingClient
from src.core.interventions import create_intervention
from src.core.init_parser import load_init_file
from src.config import load_experiment_config

# Optional: AttractorDetector (requires hdbscan)
try:
    from src.analysis.attractors import AttractorDetector
    ATTRACTOR_AVAILABLE = True
except ImportError:
    ATTRACTOR_AVAILABLE = False


def print_header():
    """Print experiment header."""
    print("=" * 60)
    print("LOGOSPHERE")
    print("Memetic dynamics experiment")
    print("=" * 60)
    print()


def create_from_template(name: str, template_name: str, model: str = None) -> None:
    """
    Create new experiment from template experiment.

    Copies config.json and init.md from template to new experiment directory.
    """
    import shutil

    template_dir = config.EXPERIMENTS_DIR / template_name
    new_dir = config.EXPERIMENTS_DIR / name

    if not template_dir.exists():
        print(f"❌ Error: Template experiment not found")
        print(f"Expected: {template_dir}")
        sys.exit(1)

    if new_dir.exists():
        print(f"❌ Error: Experiment directory already exists")
        print(f"Location: {new_dir}")
        sys.exit(1)

    new_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created: {new_dir}")

    # Copy config.json
    template_config = template_dir / "config.json"
    new_config = new_dir / "config.json"
    if template_config.exists():
        shutil.copy2(template_config, new_config)
        print(f"Copied: config.json from {template_name}")

    # Copy init.md
    template_init = template_dir / "init.md"
    new_init = new_dir / "init.md"
    if template_init.exists():
        shutil.copy2(template_init, new_init)
        print(f"Copied: init.md from {template_name}")

    # Override model if specified
    if model and new_config.exists():
        with open(new_config, 'r') as f:
            cfg = json.load(f)
        cfg['api']['MODEL'] = model
        with open(new_config, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f"Updated: MODEL={model}")

    print()
    print(f"✓ Experiment '{name}' created from template '{template_name}'")
    print()


def run_experiment(name: str, analyze: bool = True, model: str = None) -> None:
    """
    Run Logosphere experiment with VectorDB and real-time detection.
    """
    print_header()

    # Verify experiment directory exists
    exp_dir = config.EXPERIMENTS_DIR / name
    if not exp_dir.exists():
        print(f"❌ Error: Experiment directory not found")
        print(f"Expected: {exp_dir}")
        print()
        print("Create experiment first:")
        print(f"  mkdir -p {exp_dir}")
        print(f"  cp config-template.json {exp_dir}/config.json")
        print(f"  cp init-template.txt {exp_dir}/init.md")
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

    # Load experiment config (with defaults merged)
    print("Loading config.json...")
    exp_config = load_experiment_config(exp_dir)
    params = exp_config.get('parameters', {})
    api = exp_config.get('api', {})
    embedding_config = exp_config.get('embeddings', {})
    attractor_config = exp_config.get('attractor_detection', {})
    intervention_config = exp_config.get('interventions', {})

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
    print(f"✓ Embeddings: {'enabled' if embedding_config.get('enabled') else 'disabled'}")
    print(f"✓ Attractors: {'enabled' if attractor_config.get('enabled') else 'disabled'}")
    print()

    # Check for init.md
    init_path = exp_dir / "init.md"
    if not init_path.exists():
        print(f"❌ Error: init.md not found")
        print(f"Expected: {init_path}")
        sys.exit(1)

    # Load init.md
    print(f"Loading init.md...")
    seed_messages = load_init_file(init_path)
    print(f"✓ Loaded {len(seed_messages)} seed messages")
    print()

    # Initialize VectorDB
    vector_db = VectorDB(active_pool_size=M_ACTIVE_POOL)

    # Initialize embedding client
    embeddings_enabled = embedding_config.get('enabled', False)
    embedding_client = EmbeddingClient(
        model=embedding_config.get('model', 'openai/text-embedding-3-small'),
        enabled=embeddings_enabled,
    )

    # Embed and add seed messages
    if seed_messages:
        if embeddings_enabled:
            print("Embedding seed messages...")
            embeddings = embedding_client.embed_batch(seed_messages)
            for msg, emb in zip(seed_messages, embeddings):
                vector_db.add(
                    text=msg,
                    embedding=emb,
                    round_num=0,
                    mind_id=-1,  # -1 indicates seed message
                )
            print(f"✓ Embedded and added {len(seed_messages)} seeds")
        else:
            # No embeddings - add with zero vectors (search won't work)
            import numpy as np
            zero_emb = np.zeros(1536, dtype=np.float32)
            for msg in seed_messages:
                vector_db.add(
                    text=msg,
                    embedding=zero_emb,
                    round_num=0,
                    mind_id=-1,
                )
            print(f"✓ Added {len(seed_messages)} seeds (no embeddings)")
        print()

    print(f"VectorDB initialized:")
    print(f"  Total messages: {vector_db.size()}")
    print(f"  Active pool: {vector_db.active_size()} (tail {M_ACTIVE_POOL})")
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
                "embeddings": embedding_config,
                "attractor_detection": attractor_config,
                "interventions": intervention_config,
            },
            init_signature="",
            num_seeds=len(seed_messages)
        )

        # Initialize attractor detector (optional)
        attractor_detector = None
        if attractor_config.get('enabled') and ATTRACTOR_AVAILABLE:
            attractor_detector = AttractorDetector(
                vector_db=vector_db,
                min_cluster_size=attractor_config.get('min_cluster_size', 5),
            )
            print(f"✓ Attractor detection enabled (min_cluster_size={attractor_config.get('min_cluster_size', 5)})")
        elif attractor_config.get('enabled') and not ATTRACTOR_AVAILABLE:
            print(f"⚠ Attractor detection requested but hdbscan not available")

        # Initialize intervention (optional)
        intervention = None
        if intervention_config.get('enabled'):
            intervention = create_intervention(intervention_config, vector_db)
            print(f"✓ Intervention: {intervention_config.get('strategy', 'none')}")

        # Create orchestrator
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
        print(f"Running {MAX_ROUNDS} rounds with {N_MINDS} Minds...")
        print()

        completed_rounds = 0
        try:
            for round_num in range(1, MAX_ROUNDS + 1):
                print(f"Round {round_num}/{MAX_ROUNDS}...", end=" ", flush=True)
                messages_added = orchestrator.run_round(round_num)
                completed_rounds = round_num
                print(f"✓ ({messages_added} messages, {vector_db.size()} total)")

            print()
            print("=" * 60)
            print("EXPERIMENT COMPLETE")
            print("=" * 60)
            print()
            print(f"Rounds completed: {MAX_ROUNDS}")
            print(f"Final pool size: {vector_db.size()}")
            print(f"Total tokens: {orchestrator.total_tokens:,}")
            print()

            # Log experiment end
            logger.log_experiment_end(
                total_rounds=MAX_ROUNDS,
                final_pool_size=vector_db.size(),
                total_tokens=orchestrator.total_tokens
            )

            # Save final VectorDB
            orchestrator.save()
            print(f"VectorDB saved to: {exp_dir / 'vector_db'}")

            print(f"Logs: {log_dir / 'experiment.jsonl'}")
            print()

            # Run analysis if requested
            if analyze:
                print("Running analysis...")
                import analyze as analyzer
                try:
                    analyzer.analyze_novel_memes(exp_dir)
                except Exception as e:
                    print(f"Warning: Analysis failed: {e}")
                    print(f"You can re-run analysis with: python analyze.py {name}")

        except ExperimentAbortError as e:
            print(f"\n\n❌ EXPERIMENT ABORTED: {e}")
            print()
            print(f"Completed {completed_rounds}/{MAX_ROUNDS} rounds")
            print(f"Current pool size: {vector_db.size()}")
            print(f"Tokens used: {orchestrator.total_tokens:,}")
            print()
            print(f"Logs: {log_dir / 'experiment.jsonl'}")

            # Save partial VectorDB
            orchestrator.save()
            print(f"VectorDB saved: {exp_dir / 'vector_db'}")
            print()
            sys.exit(1)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("EXPERIMENT INTERRUPTED")
            print("=" * 60)
            print()
            print(f"Completed {completed_rounds}/{MAX_ROUNDS} rounds")
            print(f"Current pool size: {vector_db.size()}")
            print(f"Tokens used: {orchestrator.total_tokens:,}")
            print()

            # Save partial VectorDB
            orchestrator.save()
            print(f"VectorDB saved: {exp_dir / 'vector_db'}")
            print(f"Logs: {log_dir / 'experiment.jsonl'}")
            print()
            sys.exit(1)


def main():
    """Parse arguments and run experiment."""
    parser = argparse.ArgumentParser(
        description="Run Logosphere memetic dynamics experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  # Quick model comparison
  python run.py sonnet-run --template _baseline --model anthropic/claude-sonnet-4.5
  python run.py haiku-run --template _baseline --model anthropic/claude-haiku-4.5

  # Create from template for manual editing
  python run.py new-experiment --template existing-experiment
  # Edit config.json and init.md as needed
  python run.py new-experiment

  # Create from scratch
  mkdir experiments/my-experiment
  cp config-template.json experiments/my-experiment/config.json
  cp init-template.txt experiments/my-experiment/init.md
  python run.py my-experiment
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
        help='Override MODEL in config.json'
    )

    args = parser.parse_args()

    # If template specified, create from template
    if args.template:
        create_from_template(name=args.name, template_name=args.template, model=args.model)
        if not args.model:
            print("Next steps:")
            print(f"  1. Edit experiments/{args.name}/config.json and init.md")
            print(f"  2. Run: python run.py {args.name}")
            sys.exit(0)

    run_experiment(name=args.name, analyze=args.analyze, model=args.model)


if __name__ == "__main__":
    main()
