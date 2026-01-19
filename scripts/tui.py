#!/usr/bin/env python3
"""
Mind TUI - Textual interface for Logosphere v2.

Usage:
    python scripts/tui.py              # Use session from ~/.mind_session
    python scripts/tui.py ./session    # Use specific session directory
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tui import run_tui


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mind TUI - Textual interface for Logosphere v2"
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        nargs="?",
        help="Session directory (default: use ~/.mind_session)",
    )
    args = parser.parse_args()

    try:
        run_tui(args.session_dir)
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
