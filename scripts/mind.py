#!/usr/bin/env python3
"""
Mind CLI - Logosphere v2 entrypoint.

Usage:
    mind init ./session "initial prompt"   # Create new session
    mind open ./session                    # Open existing session
    mind status                            # Show current session state
    mind run 10                            # Run N iterations
    mind step                              # Single iteration
    mind message "text"                    # Send message to mind
    mind message -f prompt.md              # Send message from file
    cat prompt.md | mind message           # Send message via pipe
    mind cluster status                    # Show cluster state
    mind cluster show cluster_0            # Show cluster members
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.session_v2 import SessionV2, SessionConfig
from src.mind import MindRunner, MindConfig


# Session tracking file
SESSION_FILE = Path.home() / ".mind_session"


def get_current_session_dir() -> Path:
    """Get the current session directory."""
    if SESSION_FILE.exists():
        return Path(SESSION_FILE.read_text().strip())
    raise RuntimeError("No session open. Use 'mind open' first.")


def set_current_session(path: Path) -> None:
    """Set the current session directory."""
    SESSION_FILE.write_text(str(path.resolve()))


# ============================================================================
# Commands
# ============================================================================

def cmd_init(args) -> int:
    """Initialize a new session."""
    session_dir = Path(args.session_dir).resolve()

    if SessionV2.exists(session_dir):
        print(f"Session already exists at {session_dir}")
        return 1

    config = SessionConfig()
    session = SessionV2.create(
        session_dir=session_dir,
        initial_prompt=args.prompt if args.prompt else None,
        config=config,
    )

    set_current_session(session_dir)
    print(f"Created session at {session_dir}")

    if args.prompt:
        print(f"Initial prompt: {args.prompt[:60]}...")

    return 0


def cmd_open(args) -> int:
    """Open an existing session."""
    session_dir = Path(args.session_dir).resolve()

    if not SessionV2.exists(session_dir):
        print(f"No session found at {session_dir}")
        return 1

    set_current_session(session_dir)
    session = SessionV2(session_dir)

    print(f"Opened session at {session_dir}")
    print(f"  Iteration: {session.iteration}")
    print(f"  Thoughts: {session.thinking_pool.size()}")
    print(f"  Messages: {session.message_pool.size()}")

    return 0


def cmd_status(args) -> int:
    """Show current session status."""
    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    print(f"Session: {session_dir}")
    print(f"  Iteration: {session.iteration}")
    print(f"  Thoughts: {session.thinking_pool.size()} (active: {session.thinking_pool.active_size()})")
    print(f"  Messages: {session.message_pool.size()}")
    print(f"  Model: {session.config.model}")

    # Show recent messages
    messages = session.get_messages()
    if messages:
        print("\nRecent messages:")
        for msg in messages[-5:]:
            preview = msg.text[:60] + "..." if len(msg.text) > 60 else msg.text
            print(f"  [{msg.source} → {msg.to}] {preview}")

    return 0


def cmd_run(args) -> int:
    """Run iterations."""
    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    config = MindConfig(
        verbose=not args.quiet,
    )

    runner = MindRunner(session, config)

    try:
        if args.iterations is not None:
            # Fixed number of iterations
            results = runner.run(args.iterations)
        else:
            # Default: run until message emitted
            results = runner.run_until_message(max_iterations=args.max)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_step(args) -> int:
    """Run single iteration."""
    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    config = MindConfig(verbose=True, debug=args.debug)
    runner = MindRunner(session, config)

    try:
        result = runner.step()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_message(args) -> int:
    """Send a message to the mind."""
    # Determine message text from: positional arg > -f file > stdin
    text = None

    if args.text:
        text = args.text
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {args.file}")
            return 1
        text = file_path.read_text()
    elif not sys.stdin.isatty():
        # Stdin is being piped
        text = sys.stdin.read()
    else:
        print("Usage: mind message \"text\" | mind message -f file.txt | echo \"text\" | mind message")
        return 1

    text = text.strip()
    if not text:
        print("Empty message")
        return 1

    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    session.add_message(
        source='user',
        to=args.to,
        text=text,
    )
    session.save()

    # Show preview for long messages
    if len(text) > 80:
        preview = text[:77] + "..."
        print(f"Message sent to {args.to} ({len(text)} chars): {preview}")
    else:
        print(f"Message sent to {args.to}: {text}")

    return 0


def cmd_cluster(args) -> int:
    """Cluster management commands."""
    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    config = MindConfig(verbose=True)
    runner = MindRunner(session, config)

    if args.cluster_command == 'status':
        status = runner.cluster_mgr.get_status()
        if not runner.cluster_mgr.initialized:
            print("Clustering not yet initialized (will auto-initialize on first run).")
            return 0

        print(f"Clusters: {status.get('num_clusters', 0)}")
        print(f"Assigned: {status.get('total_assigned', 0)}")
        print(f"Noise: {status.get('noise', 0)}")

        if status.get('clusters'):
            print("\nCluster details:")
            for cluster in status['clusters']:
                cid = cluster.get('id', '?')
                members = cluster.get('members', 0)
                rep = cluster.get('representative', '')
                print(f"  {cid}: {members} members")
                if rep:
                    print(f"    └─ {rep}")

        return 0

    elif args.cluster_command == 'show':
        if not args.cluster_id:
            print("Specify cluster ID: mind cluster show cluster_0")
            return 1

        members = runner.cluster_mgr.get_cluster_members(
            args.cluster_id,
            runner._make_clustering_adapter(),
        )
        if not members:
            print(f"No members found for {args.cluster_id}")
            return 0

        print(f"Cluster {args.cluster_id} ({len(members)} members):")
        for m in members[:10]:
            preview = m['text'][:80] + "..." if len(m['text']) > 80 else m['text']
            print(f"  [{m['vector_id']}] {preview}")

        if len(members) > 10:
            print(f"  ... and {len(members) - 10} more")

        return 0

    return 0


def cmd_config(args) -> int:
    """Show or set config."""
    import json

    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    if args.json:
        print(json.dumps(session.config.to_dict(), indent=2))
    elif args.set:
        for kv in args.set:
            if '=' not in kv:
                print(f"Invalid format: {kv} (expected key=value)")
                return 1
            key, value = kv.split('=', 1)
            # Try to parse as JSON for numbers/bools
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            if hasattr(session.config, key):
                setattr(session.config, key, value)
            else:
                print(f"Unknown config key: {key}")
                return 1
        session._save()
        print("Config updated")
    else:
        print("Config:")
        for key, value in session.config.to_dict().items():
            print(f"  {key}: {value}")

    return 0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mind CLI - Logosphere v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # init
    p_init = subparsers.add_parser('init', help='Initialize new session')
    p_init.add_argument('session_dir', help='Session directory path')
    p_init.add_argument('prompt', nargs='?', help='Initial prompt')

    # open
    p_open = subparsers.add_parser('open', help='Open existing session')
    p_open.add_argument('session_dir', help='Session directory path')

    # status
    subparsers.add_parser('status', help='Show session status')

    # run
    p_run = subparsers.add_parser('run', help='Run until message (or N iterations)')
    p_run.add_argument('iterations', type=int, nargs='?', default=None,
                       help='Number of iterations (default: run until message)')
    p_run.add_argument('--max', type=int, default=100,
                       help='Max iterations when running until message (default: 100)')
    p_run.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')

    # step
    p_step = subparsers.add_parser('step', help='Single iteration')
    p_step.add_argument('--debug', action='store_true', help='Dump full LLM request/response')

    # message
    p_msg = subparsers.add_parser('message', help='Send message to mind')
    p_msg.add_argument('text', nargs='?', help='Message text (or use -f or pipe)')
    p_msg.add_argument('-f', '--file', help='Read message from file')
    p_msg.add_argument('--to', default='mind_0', help='Recipient (default: mind_0)')

    # cluster
    p_cluster = subparsers.add_parser('cluster', help='Cluster management')
    p_cluster.add_argument('cluster_command', choices=['status', 'show'],
                           help='Cluster subcommand')
    p_cluster.add_argument('cluster_id', nargs='?', help='Cluster ID (for show)')

    # config
    p_config = subparsers.add_parser('config', help='Show/set config')
    p_config.add_argument('--set', nargs='+', metavar='KEY=VALUE',
                          help='Set config values')
    p_config.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        'init': cmd_init,
        'open': cmd_open,
        'status': cmd_status,
        'run': cmd_run,
        'step': cmd_step,
        'message': cmd_message,
        'cluster': cmd_cluster,
        'config': cmd_config,
    }

    try:
        return commands[args.command](args)
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
