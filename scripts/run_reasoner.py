#!/usr/bin/env python3
"""
Run the working memory reasoner.

Pool state = output. Minimal protocols.

Usage:
    python scripts/run_reasoner.py "Explore this topic"
    python scripts/run_reasoner.py "First thought" "Second thought" -o ./out
    python scripts/run_reasoner.py --resume ./out "New prompt to inject"
    python scripts/run_reasoner.py --resume ./out -n 10
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning import Reasoner, ReasonerConfig


def main():
    parser = argparse.ArgumentParser(
        description="Working Memory Reasoner - Pool-based thought ecology"
    )
    parser.add_argument(
        "prompts",
        nargs="*",
        help="Prompt(s) to seed the pool (or inject on resume)"
    )
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=30,
        help="Maximum reasoning iterations (default: 30)"
    )
    parser.add_argument(
        "--k-samples", "-k",
        type=int,
        default=5,
        help="Thoughts sampled per iteration (default: 5)"
    )
    parser.add_argument(
        "--pool-size", "-m",
        type=int,
        default=50,
        help="Working memory capacity (default: 50)"
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.75,
        help="Coherence threshold for convergence (default: 0.75)"
    )
    parser.add_argument(
        "--stability-window",
        type=int,
        default=3,
        help="Iterations of stable cluster count to terminate (default: 3)"
    )
    parser.add_argument(
        "--early-termination",
        action="store_true",
        help="Enable early termination on convergence/stability (default: off)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-haiku-4.5",
        help="Model to use (default: anthropic/claude-haiku-4.5)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Show detailed output (default: True)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Save VectorDB to this directory"
    )
    parser.add_argument(
        "--resume", "-r",
        type=Path,
        help="Resume from a prior run's output directory"
    )

    args = parser.parse_args()

    prompts = args.prompts or []

    # Handle resume vs fresh start
    if args.resume:
        # Resume from checkpoint
        try:
            reasoner = Reasoner.from_checkpoint(args.resume)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Build config overrides from CLI args
        # -n is additive on resume: adds iterations to current position
        overrides = {
            "max_iterations": reasoner.iteration + args.max_iterations,
            "no_early_termination": not args.early_termination,
            "verbose": not args.quiet,
        }
        if args.k_samples != 5:
            overrides["k_samples"] = args.k_samples
        if args.pool_size != 50:
            overrides["active_pool_size"] = args.pool_size
        if args.convergence_threshold != 0.75:
            overrides["convergence_threshold"] = args.convergence_threshold
        if args.stability_window != 3:
            overrides["stability_window"] = args.stability_window
        if args.model != "anthropic/claude-haiku-4.5":
            overrides["model"] = args.model

        # Apply overrides
        for key, value in overrides.items():
            setattr(reasoner.config, key, value)

        print("=" * 60)
        print("WORKING MEMORY REASONER (RESUME)")
        print(f"Resuming from: {args.resume}")
        if prompts:
            print(f"Injecting {len(prompts)} new prompt(s)")
        print(f"Model: {reasoner.config.model}")
        print("=" * 60)

        result = reasoner.continue_run(prompts if prompts else None)
    else:
        # Fresh start - require at least one prompt
        if not prompts:
            parser.print_help()
            print("\nError: Please provide at least one prompt (or use --resume)")
            sys.exit(1)

        # Configure
        config = ReasonerConfig(
            max_iterations=args.max_iterations,
            k_samples=args.k_samples,
            active_pool_size=args.pool_size,
            convergence_threshold=args.convergence_threshold,
            stability_window=args.stability_window,
            no_early_termination=not args.early_termination,
            model=args.model,
            verbose=not args.quiet,
            output_dir=args.output_dir
        )

        # Run
        reasoner = Reasoner(config)

        print("=" * 60)
        print("WORKING MEMORY REASONER")
        print(f"Model: {config.model}")
        print("=" * 60)

        result = reasoner.run(prompts)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Termination: {result.termination_reason}")
    print(f"Iterations: {result.iterations}")
    print(f"Total thoughts: {result.total_thoughts}")
    print(f"Final pool size: {result.final_pool_size}")
    print(f"Clusters: {result.cluster_count}")
    print(f"Final coherence: {result.final_coherence:.3f}")

    print("\n--- Dominant cluster representatives ---")
    for i, text in enumerate(result.dominant_cluster_texts, 1):
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"{i}. {preview}")

    # Print metrics trajectory if verbose
    if not args.quiet and result.metrics_history:
        print("\n--- Metrics trajectory ---")
        print("iter  clusters  coherence  diversity  share")
        for m in result.metrics_history:
            print(f"{m.iteration:4d}  {m.num_clusters:8d}  {m.coherence:9.3f}  "
                  f"{m.diversity:9.3f}  {m.dominant_cluster_share:.3f}")

    if reasoner.config.output_dir:
        print(f"\nSaved to: {reasoner.config.output_dir}")


if __name__ == "__main__":
    main()
