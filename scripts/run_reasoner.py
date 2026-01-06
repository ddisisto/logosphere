#!/usr/bin/env python3
"""
Run the working memory reasoner on a problem.

No protocols. Pool state = output. Termination is dynamics-based.

Usage:
    python scripts/run_reasoner.py "What is 23 + 47?"
    python scripts/run_reasoner.py --max-iterations 20 "problem here"
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
        "problem",
        nargs="?",
        help="The problem to solve"
    )
    parser.add_argument(
        "--problem", "-p",
        dest="problem_flag",
        help="The problem to solve (alternative to positional)"
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

    args = parser.parse_args()

    # Get problem from either source
    problem = args.problem or args.problem_flag
    if not problem:
        parser.print_help()
        print("\nError: Please provide a problem to solve")
        sys.exit(1)

    # Configure
    config = ReasonerConfig(
        max_iterations=args.max_iterations,
        k_samples=args.k_samples,
        active_pool_size=args.pool_size,
        convergence_threshold=args.convergence_threshold,
        verbose=not args.quiet,
        output_dir=args.output_dir
    )

    # Run
    reasoner = Reasoner(config)

    print("=" * 60)
    print("WORKING MEMORY REASONER")
    print("Pool state = output. No protocols.")
    print("=" * 60)

    result = reasoner.solve(problem)

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

    if args.output_dir:
        print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
