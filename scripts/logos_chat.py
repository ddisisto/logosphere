#!/usr/bin/env python3
"""
Logos Chat - Interactive TUI for structured exchange.

Usage:
    python scripts/logos_chat.py ./session-dir
    python scripts/logos_chat.py  # Uses current session from ~/.logos_session
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tui.chat import run_chat


def get_current_session_dir() -> Path:
    """Get the current session directory."""
    session_file = Path.home() / ".logos_session"
    if session_file.exists():
        return Path(session_file.read_text().strip())
    raise RuntimeError("No session open. Use 'logos open' or specify session directory.")


def main():
    parser = argparse.ArgumentParser(
        description="Logos Chat - Interactive TUI for structured exchange"
    )
    parser.add_argument(
        "session_dir",
        nargs="?",
        help="Session directory (default: current session)",
    )

    args = parser.parse_args()

    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        session_dir = get_current_session_dir()

    if not session_dir.exists():
        print(f"Session not found: {session_dir}")
        return 1

    if not (session_dir / "session.json").exists():
        print(f"Not a valid session directory: {session_dir}")
        print("(Missing session.json)")
        return 1

    print(f"Opening chat for session: {session_dir}")
    run_chat(session_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
