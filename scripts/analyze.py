#!/usr/bin/env python3
"""
Analysis tools for Logosphere experiments.

Usage:
    python analyze.py <experiment-name> [--tool <tool-name>]

Available tools:
    novel-memes (default) - Extract novel memes to YAML

To add new tools:
    1. Define function: def analyze_<toolname>(exp_dir: Path) -> None
    2. Add to TOOLS registry
    3. That's it!
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config

# Optional imports for embeddings tool
try:
    import numpy as np
    import requests
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


# Analysis Tool Registry
# To add new tool: define analyze_<name> function and add to this dict
TOOLS: dict[str, Callable[[Path], None]] = {}


def analyze_novel_memes(exp_dir: Path) -> None:
    """
    Extract novel memes (messages transmitted during experiment) to YAML.

    Reads: experiments/<name>/logs/experiment.jsonl
    Writes: experiments/<name>/novel_memes.yaml

    Format:
        ---
        round: N
        mind_id: M
        content: |-
          Message text here
    """
    log_path = exp_dir / "logs" / "experiment.jsonl"
    output_path = exp_dir / "novel_memes.yaml"

    if not log_path.exists():
        print(f"Error: experiment.jsonl not found")
        print(f"Expected: {log_path}")
        sys.exit(1)

    print(f"Reading: {log_path}")

    # Extract novel memes from mind_invocation events
    novel_memes = []
    with open(log_path) as f:
        for line in f:
            event = json.loads(line)
            if event['type'] == 'mind_invocation':
                for msg in event['transmitted']:
                    novel_memes.append({
                        'round': event['round'],
                        'mind_id': event['mind_id'],
                        'content': msg
                    })

    print(f"Found: {len(novel_memes)} novel memes")

    # Write to YAML
    with open(output_path, 'w') as f:
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

    print(f"Wrote: {output_path}")
    print()


def analyze_embeddings(exp_dir: Path) -> None:
    """
    Generate embeddings and diversity analysis for experiment messages.

    Computes diversity metrics over time and generates visualizations.
    Requires: numpy, requests, sklearn, matplotlib

    Reads: init.md, logs/experiment.jsonl
    Writes: embeddings.npz, diversity_metrics.json, diversity_plots.png
    """
    if not EMBEDDINGS_AVAILABLE:
        print("Error: Missing dependencies for embeddings analysis")
        print("Install with: pip install numpy requests scikit-learn matplotlib")
        sys.exit(1)

    # Load API key from .env
    try:
        api_key = config.load_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        print("Ensure .env file exists with OPENROUTER_API key")
        sys.exit(1)

    print("Loading messages...")
    messages, metadata = _load_all_messages(exp_dir)
    print(f"  Loaded {len(messages)} messages")
    print(f"    Seed messages: {metadata['num_seeds']}")
    print(f"    Novel messages: {len(messages) - metadata['num_seeds']}")
    print()

    # Check for cached embeddings
    embeddings_path = exp_dir / "embeddings.npz"
    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}...")
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        cached_messages = data['messages'].tolist()

        # Verify cache is valid
        if cached_messages == messages:
            print("  ✓ Cache valid")
        else:
            print("  ⚠ Cache invalid (messages changed), regenerating...")
            embeddings = _generate_embeddings(messages, api_key)
            _save_embeddings(embeddings_path, embeddings, messages)
    else:
        print("Generating embeddings...")
        embeddings = _generate_embeddings(messages, api_key)
        _save_embeddings(embeddings_path, embeddings, messages)

    print()

    # Compute diversity metrics
    print("Computing diversity metrics...")
    metrics = _compute_diversity_metrics(embeddings, metadata)

    # Save metrics
    metrics_path = exp_dir / "diversity_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Wrote: {metrics_path}")
    print()

    # Generate visualizations
    print("Generating visualizations...")
    plots_path = exp_dir / "diversity_plots.png"
    _generate_diversity_plots(metrics, metadata, plots_path)
    print(f"  Wrote: {plots_path}")
    print()


def _load_all_messages(exp_dir: Path) -> tuple[list[str], dict]:
    """
    Load seed messages and novel messages in chronological order.

    Returns:
        messages: List of all messages (seeds first, then novel by round)
        metadata: Dict with num_seeds, rounds info
    """
    messages = []

    # Load seed messages from init.md
    init_path = exp_dir / "init.md"
    if init_path.exists():
        from src.core.init_parser import load_init_file
        seed_messages = load_init_file(init_path)
        messages.extend(seed_messages)
    else:
        print(f"Warning: init.md not found at {init_path}")
        seed_messages = []

    num_seeds = len(messages)

    # Load novel messages from experiment.jsonl
    log_path = exp_dir / "logs" / "experiment.jsonl"
    if not log_path.exists():
        print(f"Error: experiment.jsonl not found at {log_path}")
        sys.exit(1)

    round_messages = {}  # round -> list of messages
    with open(log_path) as f:
        for line in f:
            event = json.loads(line)
            if event['type'] == 'mind_invocation':
                round_num = event['round']
                if round_num not in round_messages:
                    round_messages[round_num] = []
                round_messages[round_num].extend(event['transmitted'])

    # Add messages in round order
    for round_num in sorted(round_messages.keys()):
        messages.extend(round_messages[round_num])

    metadata = {
        'num_seeds': num_seeds,
        'num_rounds': len(round_messages),
        'round_messages': {r: len(msgs) for r, msgs in round_messages.items()}
    }

    return messages, metadata


def _generate_embeddings(messages: list[str], api_key: str) -> np.ndarray:
    """
    Generate embeddings for messages using OpenRouter API.

    Uses batch API calls for efficiency.
    """
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Batch size (OpenRouter should handle this, but let's be conservative)
    batch_size = 50  # Smaller batches to avoid rate limits
    all_embeddings = []
    import time

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]

        # Filter out empty strings (API rejects them)
        non_empty_indices = [idx for idx, msg in enumerate(batch) if msg.strip()]
        non_empty_batch = [batch[idx] for idx in non_empty_indices]

        print(f"  Batch {i//batch_size + 1}/{(len(messages)-1)//batch_size + 1} ({len(batch)} messages, {len(non_empty_batch)} non-empty)...", end=" ", flush=True)

        # Skip if entire batch is empty
        if not non_empty_batch:
            print("⊘ (all empty)")
            # Add zero vectors for empty messages (dimension determined from first batch)
            if all_embeddings:
                embedding_dim = len(all_embeddings[0])
                all_embeddings.extend([np.zeros(embedding_dim) for _ in batch])
            continue

        data = {
            "model": "openai/text-embedding-3-small",
            "input": non_empty_batch,
            "encoding_format": "float"
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()

            # Check for error response
            if 'error' in result:
                print(f"\nAPI Error: {result['error']}")
                print(f"Retrying batch...")
                # Retry once
                import time
                time.sleep(2)
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                if 'error' in result:
                    raise ValueError(f"API error after retry: {result['error']}")

            # Check response structure
            if 'data' not in result:
                raise ValueError(f"API response missing 'data' field: {result.keys()}")

            non_empty_embeddings = [item['embedding'] for item in result['data']]

            # Reconstruct full batch with zero vectors for empty strings
            embedding_dim = len(non_empty_embeddings[0])
            batch_embeddings = []
            non_empty_iter = iter(non_empty_embeddings)

            for idx in range(len(batch)):
                if idx in non_empty_indices:
                    batch_embeddings.append(next(non_empty_iter))
                else:
                    # Empty string - use zero vector
                    batch_embeddings.append(np.zeros(embedding_dim))

            all_embeddings.extend(batch_embeddings)

            print("✓")

            # Small delay to avoid rate limiting
            if i + batch_size < len(messages):
                time.sleep(0.5)

        except Exception as e:
            print(f"\nError processing batch: {e}")
            print(f"Batch preview: {batch[:2]}")
            raise

    return np.array(all_embeddings)


def _save_embeddings(path: Path, embeddings: np.ndarray, messages: list[str]) -> None:
    """Save embeddings and messages to npz file for caching."""
    np.savez_compressed(
        path,
        embeddings=embeddings,
        messages=np.array(messages, dtype=object)
    )
    print(f"  Cached embeddings to {path}")


def _compute_diversity_metrics(embeddings: np.ndarray, metadata: dict) -> dict:
    """
    Compute diversity metrics over time.

    Returns dict with time series of:
    - mean_similarity: Average pairwise similarity within each round
    - distance_from_seed: Mean distance from seed centroid
    """
    num_seeds = metadata['num_seeds']
    round_messages = metadata['round_messages']

    # Compute seed centroid
    seed_embeddings = embeddings[:num_seeds]
    seed_centroid = seed_embeddings.mean(axis=0, keepdims=True)

    # Track metrics per round
    metrics = {
        'rounds': [],
        'mean_similarity': [],  # Within-round average similarity
        'distance_from_seed': [],  # Distance from seed centroid
        'num_messages': []
    }

    # Process each round
    idx = num_seeds  # Start after seeds
    for round_num in sorted(round_messages.keys()):
        num_msgs = round_messages[round_num]
        if num_msgs == 0:
            continue

        round_embeds = embeddings[idx:idx + num_msgs]

        # Mean pairwise similarity within round
        if num_msgs > 1:
            similarities = cosine_similarity(round_embeds)
            # Get upper triangle (excluding diagonal)
            upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
            mean_sim = float(upper_tri.mean())
        else:
            mean_sim = 1.0  # Single message is perfectly similar to itself

        # Distance from seed centroid
        distances = 1 - cosine_similarity(round_embeds, seed_centroid).flatten()
        mean_dist = float(distances.mean())

        metrics['rounds'].append(round_num)
        metrics['mean_similarity'].append(mean_sim)
        metrics['distance_from_seed'].append(mean_dist)
        metrics['num_messages'].append(num_msgs)

        idx += num_msgs

    return metrics


def _generate_diversity_plots(metrics: dict, metadata: dict, output_path: Path) -> None:
    """Generate diversity visualization plots."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    rounds = metrics['rounds']

    # Plot 1: Messages per round
    axes[0].plot(rounds, metrics['num_messages'], 'o-', linewidth=2, markersize=4)
    axes[0].set_ylabel('Messages per round', fontsize=12)
    axes[0].set_title('Output Volume Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Mean similarity (diversity inverse)
    axes[1].plot(rounds, metrics['mean_similarity'], 'o-', linewidth=2, markersize=4, color='orange')
    axes[1].set_ylabel('Mean pairwise similarity', fontsize=12)
    axes[1].set_title('Within-Round Diversity (lower = more diverse)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    # Plot 3: Distance from seed
    axes[2].plot(rounds, metrics['distance_from_seed'], 'o-', linewidth=2, markersize=4, color='green')
    axes[2].set_xlabel('Round', fontsize=12)
    axes[2].set_ylabel('Distance from seed centroid', fontsize=12)
    axes[2].set_title('Drift from Initial Conditions', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# Register tools
TOOLS['novel-memes'] = analyze_novel_memes
TOOLS['embeddings'] = analyze_embeddings


def list_tools() -> None:
    """Print available analysis tools."""
    print("Available analysis tools:")
    print()
    for name, func in TOOLS.items():
        # Extract first line of docstring
        doc = func.__doc__ or ""
        first_line = doc.strip().split('\n')[0] if doc else "No description"
        print(f"  {name:15} - {first_line}")
    print()


def main():
    """Parse arguments and run analysis tool."""
    parser = argparse.ArgumentParser(
        description="Analyze Logosphere experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract novel memes (default tool)
  python analyze.py my-experiment

  # Explicitly specify tool
  python analyze.py my-experiment --tool novel-memes

  # List available tools
  python analyze.py --list-tools

Analysis is idempotent - safe to run multiple times.
        """
    )

    parser.add_argument(
        'name',
        type=str,
        nargs='?',
        help='Experiment name (directory under experiments/)'
    )

    parser.add_argument(
        '--tool',
        type=str,
        default='novel-memes',
        choices=list(TOOLS.keys()),
        help='Analysis tool to run (default: novel-memes)'
    )

    parser.add_argument(
        '--list-tools',
        action='store_true',
        help='List available analysis tools and exit'
    )

    args = parser.parse_args()

    # Handle --list-tools
    if args.list_tools:
        list_tools()
        sys.exit(0)

    # Validate experiment name provided
    if not args.name:
        parser.print_help()
        print()
        print("Error: experiment name required (or use --list-tools)")
        sys.exit(1)

    # Verify experiment directory exists
    exp_dir = config.EXPERIMENTS_DIR / args.name
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found")
        print(f"Expected: {exp_dir}")
        sys.exit(1)

    print(f"Experiment: {args.name}")
    print(f"Directory: {exp_dir}")
    print(f"Tool: {args.tool}")
    print()

    # Run analysis tool
    tool_func = TOOLS[args.tool]
    tool_func(exp_dir)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
