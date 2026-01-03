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

import config


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


# Register tools
TOOLS['novel-memes'] = analyze_novel_memes


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
