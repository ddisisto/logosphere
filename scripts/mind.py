#!/usr/bin/env python3
"""
Mind CLI - Logosphere v2 entrypoint.

Usage:
    mind init ./session "initial prompt"   # Create new session
    mind open ./session                    # Open existing session
    mind status                            # Show current session state
    mind run 10                            # Run N iterations
    mind run                               # Run until draft produced
    mind step                              # Single iteration
    mind message "text"                    # Send message to mind
    mind message -f prompt.md              # Send message from file
    mind accept                            # Accept latest draft
    mind accept 247                        # Accept draft by iteration number
    mind accept -2                         # Accept second-latest draft
    mind drafts                            # Show current drafts (truncated)
    mind drafts show 247                   # Show full draft by iter
    mind drafts show -1                    # Show full draft by offset
    mind drafts seen                       # Mark all drafts as seen
    mind drafts seen 247 250               # Mark specific drafts as seen (by iter)
    mind history                           # Show all history (truncated)
    mind history -5                        # Show last 5 entries (truncated)
    mind history 247                       # Show entry by iter (full view)
    mind history -1                        # Show latest entry (full view)
    mind cluster status                    # Show cluster state
    mind cluster show cluster_0            # Show cluster members
    mind signal                            # Show current signal
    mind signal -a                         # Show all signal history
    mind signal -p reviewing               # Set presence (a/r/e shortcuts)
    mind signal -s "focusing on X"         # Set status text
    mind signal -p e -s "deep work"        # Set both
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.session_v2 import SessionV2, SessionConfig
from src.core.draft_store import Draft
from src.core.lock import SessionLock, check_lock, LockError
from src.mind import MindRunner, MindConfig, ops


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


def require_unlocked(session_dir: Path) -> bool:
    """
    Check if session is locked by another process.

    Returns True if unlocked/available, False if locked (prints error).
    """
    lock_info = check_lock(session_dir)
    if lock_info is not None:
        print(f"Error: Session locked by {lock_info.holder} (pid {lock_info.pid})")
        print("  Release the lock or wait for the other process to finish.")
        return False
    return True


def truncate_text(text: str, max_lines: int = 8) -> tuple[str, int]:
    """
    Truncate text to max_lines.

    Returns:
        (truncated_text, remaining_lines) - remaining_lines is 0 if not truncated
    """
    lines = text.split('\n')
    if len(lines) <= max_lines:
        return text, 0
    truncated = '\n'.join(lines[:max_lines])
    return truncated, len(lines) - max_lines


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

    # Show dialogue state
    if session.is_drafting:
        all_drafts = session.get_all_drafts()
        print(f"  State: drafting ({len(all_drafts)} drafts)")
    else:
        history = session.get_history()
        print(f"  State: idle ({len(history) // 2} exchanges in history)")

    return 0


def cmd_status(args) -> int:
    """Show current session status."""
    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    print(f"Session: {session_dir}")
    print(f"  Iteration: {session.iteration}")
    print(f"  Thoughts: {session.thinking_pool.size()} (active: {session.thinking_pool.active_size()})")
    print(f"  Model: {session.config.model}")

    # Show dialogue state
    if session.is_drafting:
        awaiting = session.get_awaiting_message()
        all_drafts = session.get_all_drafts()
        print(f"\nAwaiting response:")
        if awaiting:
            preview = awaiting.text[:60] + "..." if len(awaiting.text) > 60 else awaiting.text
            print(f"  {preview}")
        print(f"\nDrafts: {len(all_drafts)}")
        if all_drafts:
            latest = all_drafts[-1]
            preview = latest.text[:60] + "..." if len(latest.text) > 60 else latest.text
            seen = "seen" if latest.seen else "unseen"
            print(f"  #{latest.index} ({seen}) {preview}")
    else:
        history = session.get_history()
        print(f"\nState: idle ({len(history) // 2} exchanges in history)")
        if history:
            print("\nRecent history:")
            for entry in history[-4:]:
                role = "You" if entry.role == "user" else "Mind"
                preview = entry.text[:50] + "..." if len(entry.text) > 50 else entry.text
                print(f"  [{role}] {preview}")

    return 0


def cmd_run(args) -> int:
    """Run iterations."""
    session_dir = get_current_session_dir()

    try:
        with SessionLock(session_dir, "cli:run"):
            session = SessionV2(session_dir)

            config = MindConfig(
                verbose=not args.quiet,
            )

            runner = MindRunner(session, config)

            # -b flag controls observe mode (user presence)
            # observe=True: user watching, mark drafts seen, stop on each draft
            # observe=False: user away, don't mark seen, only stop on hard signal
            observe = not args.background

            if args.iterations is not None:
                # Fixed number of iterations
                results = runner.run(iterations=args.iterations, observe=observe)
            else:
                # Run until stop condition (default max: 100)
                results = runner.run(max_iterations=100, observe=observe)

            # Show stop reason for run-until-stop mode
            if args.iterations is None and results and not args.quiet:
                last = results[-1]
                if last.draft_added:
                    print("Stop reason: draft produced")
                elif last.hard_signal and last.thoughts_added == 0:
                    print("Stop reason: true silence")
                elif last.hard_signal:
                    print("Stop reason: hard signal (consecutive no-drafts)")
                else:
                    print("Stop reason: max iterations reached")

            return 0
    except LockError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_step(args) -> int:
    """Run single iteration."""
    session_dir = get_current_session_dir()

    try:
        with SessionLock(session_dir, "cli:step"):
            session = SessionV2(session_dir)

            config = MindConfig(verbose=True, debug=args.debug)
            runner = MindRunner(session, config)

            result = runner.step()
            return 0
    except LockError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_message(args) -> int:
    """Send a message to the mind."""
    session_dir = get_current_session_dir()
    if not require_unlocked(session_dir):
        return 1
    session = SessionV2(session_dir)

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

    result = ops.send_message(session, text or "")
    if not result.success:
        print(f"Error: {result.error}")
        if session.is_drafting and session.get_all_drafts():
            print("Use 'mind accept' to accept a draft first, or 'mind drafts' to view them.")
        return 1

    # Show preview for long messages
    if len(result.text) > 80:
        preview = result.text[:77] + "..."
        print(f"Message sent ({len(result.text)} chars): {preview}")
    else:
        print(f"Message sent: {result.text}")

    return 0


def cmd_accept(args) -> int:
    """Accept a draft response by iter number or negative offset."""
    session_dir = get_current_session_dir()
    if not require_unlocked(session_dir):
        return 1
    session = SessionV2(session_dir)

    result = ops.accept_draft(session, args.index)
    if not result.success:
        print(f"Error: {result.error}")
        if not session.is_drafting:
            pass  # Error already explains
        elif not session.get_all_drafts():
            print("Run 'mind run' to generate drafts.")
        return 1

    draft = result.draft
    print(f"Accepted draft #{draft.index} (iter {draft.iter}):")
    print("-" * 40)
    print(draft.text)
    print("-" * 40)
    print("Exchange added to history. Ready for next message.")
    return 0


def cmd_drafts(args) -> int:
    """Show or manage drafts."""
    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    if not session.is_drafting:
        print("No pending message. Send a message with 'mind message' first.")
        return 0

    all_drafts = session.get_all_drafts()

    # Handle 'show' subcommand
    if args.drafts_command == 'show':
        if not args.refs:
            print("Usage: mind drafts show <index>")
            print("  mind drafts show 0    # Show first draft")
            print("  mind drafts show 2    # Show draft #2")
            return 1

        if not all_drafts:
            print("No drafts available.")
            return 1

        index = int(args.refs[0])
        draft = next((d for d in all_drafts if d.index == index), None)
        if draft is None:
            valid_indices = [d.index for d in all_drafts]
            print(f"Draft #{index} not found.")
            print(f"Valid indices: {valid_indices}")
            return 1

        seen = "[seen]" if draft.seen else "[unseen]"
        print(f"Draft #{draft.index} (iter {draft.iter}) {seen}")
        print("-" * 40)
        print(draft.text)
        print("-" * 40)
        return 0

    # Handle 'seen' subcommand (mutating - requires unlock)
    if args.drafts_command == 'seen':
        if not require_unlocked(session_dir):
            return 1
        if not all_drafts:
            print("No drafts to mark as seen.")
            return 0

        indices = [int(r) for r in args.refs] if args.refs else None
        count, marked = ops.mark_drafts_seen(session, indices)
        if indices:
            if marked:
                print(f"Marked drafts as seen: {marked}")
            else:
                print("No matching drafts found.")
        else:
            print(f"Marked {count} draft(s) as seen.")
        return 0

    # Default: show all drafts (truncated)
    awaiting = session.get_awaiting_message()
    print("Awaiting response to:")
    print("-" * 40)
    if awaiting:
        truncated_awaiting, remaining_awaiting = truncate_text(awaiting.text, max_lines=8)
        print(truncated_awaiting)
        if remaining_awaiting > 0:
            print(f"... ({remaining_awaiting} more lines)")
    print("-" * 40)

    if not all_drafts:
        print("\nNo drafts yet. Run 'mind run' to generate drafts.")
        return 0

    print(f"\nDrafts ({len(all_drafts)} total, newest first):")
    print()

    # Show newest first
    for draft in reversed(all_drafts):
        seen = "[seen]" if draft.seen else "[unseen]"
        print(f"Draft #{draft.index} (iter {draft.iter}) {seen}")
        print("\u2500" * 20)  # Unicode box-drawing horizontal line

        # Truncate to max 8 lines
        truncated, remaining = truncate_text(draft.text, max_lines=8)
        # Show text with indentation
        for line in truncated.split('\n'):
            print(f"  {line}")
        if remaining > 0:
            print(f"  ... ({remaining} more lines)")
        print()

    latest = all_drafts[-1]
    print("Commands:")
    print(f"  mind accept              Accept latest draft (#{latest.index})")
    print(f"  mind accept <index>      Accept draft by index")
    print(f"  mind drafts show <index> Show full draft text")
    print("  mind drafts seen         Mark all as seen")

    return 0


def cmd_history(args) -> int:
    """Show conversation history."""
    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    history = session.get_history()
    if not history:
        print("No conversation history yet.")
        return 0

    # Determine which entries to show
    if args.ref is not None:
        entries = ops.resolve_history_ref(args.ref, history)
        if not entries:
            if args.ref > 0:
                # Positive ref: looking for specific iter
                valid_iters = [m.iter for m in history]
                print(f"History entry not found for iteration {args.ref}.")
                print(f"Valid iteration numbers: {valid_iters}")
            else:
                print(f"No entries found for reference {args.ref}.")
            return 1
    else:
        # Default: show all
        entries = list(history)

    # Single vs multiple: single entry shows FULL text, multiple truncates
    is_single = len(entries) == 1

    if is_single:
        msg = entries[0]
        role_label = "user" if msg.role == "user" else "mind"
        print(f"[{role_label}] iter {msg.iter}")
        print("-" * 20)
        # Full view for single entry
        for line in msg.text.split('\n'):
            print(f"  {line}")
        print("-" * 20)
    else:
        # Multiple entries: truncated view
        total_exchanges = len(history) // 2
        shown_count = len(entries)
        print(f"History ({shown_count} messages, {total_exchanges} exchanges total):")
        print()

        for msg in entries:
            role_label = "user" if msg.role == "user" else "mind"
            print(f"[{role_label}] iter {msg.iter}")
            print("-" * 20)

            # Truncate to max 8 lines
            truncated, remaining = truncate_text(msg.text, max_lines=8)
            for line in truncated.split('\n'):
                print(f"  {line}")
            if remaining > 0:
                print(f"  ... ({remaining} more lines)")
            print()

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


def cmd_signal(args) -> int:
    """Show or update user signal (presence/status)."""
    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    if args.all:
        # Show all signal history
        print("Signal history:")
        for sig in session.user_signal:
            status_str = f' - "{sig.status}"' if sig.status else ''
            print(f"  iter {sig.iter}: {sig.presence}{status_str} ({sig.time})")
        return 0

    if args.presence or args.status:
        # Update signal (requires unlocked session)
        if not require_unlocked(session_dir):
            return 1
        result = ops.set_signal(session, presence=args.presence, status=args.status)
        if not result.success:
            print(f"Error: {result.error}")
            return 1
        signal = result.signal
        status_str = f' - "{signal.status}"' if signal.status else ''
        print(f"Signal updated: {signal.presence}{status_str}")
        return 0

    # Show current signal
    sig = session.get_latest_signal()
    age = session.iteration - sig.iter
    status_str = f'\n  status: "{sig.status}"' if sig.status else ''
    print(f"Current signal (age {age}):")
    print(f"  presence: {sig.presence}{status_str}")
    print(f"  time: {sig.time}")
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
    p_run = subparsers.add_parser('run', help='Run until stop (or N iterations)')
    p_run.add_argument('iterations', type=int, nargs='?', default=None,
                       help='Number of iterations (default: run until stop, max 100)')
    p_run.add_argument('-b', '--background', action='store_true',
                       help='Background mode: user away, drafts unseen, stop on hard signal only')
    p_run.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')

    # step
    p_step = subparsers.add_parser('step', help='Single iteration')
    p_step.add_argument('--debug', action='store_true', help='Dump full LLM request/response')

    # message
    p_msg = subparsers.add_parser('message', help='Send message to mind')
    p_msg.add_argument('text', nargs='?', help='Message text (or use -f or pipe)')
    p_msg.add_argument('-f', '--file', help='Read message from file')

    # accept
    p_accept = subparsers.add_parser('accept', help='Accept a draft response')
    p_accept.add_argument('index', type=int, nargs='?', default=None,
                          help='Draft index (0-based). None = latest')

    # drafts
    p_drafts = subparsers.add_parser('drafts', help='Show or manage drafts')
    p_drafts.add_argument('drafts_command', nargs='?', choices=['seen', 'show'],
                          help='Subcommand: seen, show')
    p_drafts.add_argument('refs', nargs='*', help='Draft indices (0-based)')

    # history
    p_history = subparsers.add_parser('history', help='Show conversation history')
    p_history.add_argument('ref', type=int, nargs='?', default=None,
                           help='History entry ref: iter number or negative offset (-N=last N entries, -1=latest single entry)')

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

    # signal
    p_signal = subparsers.add_parser('signal', help='Show/update user signal')
    p_signal.add_argument('-p', '--presence', metavar='STATE',
                          help='Set presence: absent (a), reviewing (r), engaged (e)')
    p_signal.add_argument('-s', '--status', metavar='TEXT',
                          help='Set status text')
    p_signal.add_argument('-a', '--all', action='store_true',
                          help='Show all signal history')

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
        'accept': cmd_accept,
        'drafts': cmd_drafts,
        'history': cmd_history,
        'cluster': cmd_cluster,
        'config': cmd_config,
        'signal': cmd_signal,
    }

    try:
        return commands[args.command](args)
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
