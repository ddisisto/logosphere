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
    mind history                           # Show conversation history
    mind cluster status                    # Show cluster state
    mind cluster show cluster_0            # Show cluster members
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.session_v2 import SessionV2, SessionConfig
from src.core.dialogue_pool import Draft
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
# Draft resolution helpers
# ============================================================================

def resolve_draft_ref(ref: int, drafts: list[Draft]) -> Draft | None:
    """
    Resolve a draft reference to a Draft object.

    Args:
        ref: Either an iteration number (positive) or offset (negative).
             -1 is latest, -2 is second-latest, etc.
        drafts: List of drafts (oldest first)

    Returns:
        The matching Draft or None if not found
    """
    if not drafts:
        return None

    if ref < 0:
        # Negative offset: -1 is latest, -2 is second-latest
        idx = len(drafts) + ref
        if 0 <= idx < len(drafts):
            return drafts[idx]
        return None
    else:
        # Positive: match by iteration number
        for draft in drafts:
            if draft.iter == ref:
                return draft
        return None


def get_draft_offset(draft: Draft, drafts: list[Draft]) -> int:
    """
    Get the negative offset for a draft.

    Returns:
        Negative offset (-1 for latest, -2 for second-latest, etc.)
    """
    # drafts is oldest-first, so latest is at index len-1
    idx = drafts.index(draft)
    return idx - len(drafts)  # e.g., for latest: (len-1) - len = -1


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
        awaiting = session.dialogue_pool.awaiting
        all_drafts = session.get_all_drafts()
        print(f"\nAwaiting response:")
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
            # Default: run until draft produced
            results = runner.run_until_draft(max_iterations=args.max)
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
    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    # Check for pending response
    if session.is_drafting:
        all_drafts = session.get_all_drafts()
        if all_drafts:
            print(f"Error: Cannot send message while awaiting response ({len(all_drafts)} draft(s) pending).")
            print("Use 'mind accept' to accept a draft first, or 'mind drafts' to view them.")
        else:
            print("Error: Cannot send message while awaiting response.")
            print("Run 'mind run' to generate a draft, then 'mind accept' to accept it.")
        return 1

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

    try:
        session.send_message(text)
        session.save()
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    # Show preview for long messages
    if len(text) > 80:
        preview = text[:77] + "..."
        print(f"Message sent ({len(text)} chars): {preview}")
    else:
        print(f"Message sent: {text}")

    return 0


def cmd_accept(args) -> int:
    """Accept a draft response by iter number or negative offset."""
    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    if not session.is_drafting:
        print("No pending drafts to accept.")
        return 1

    all_drafts = session.get_all_drafts()
    if not all_drafts:
        print("No drafts available. Run 'mind run' to generate drafts.")
        return 1

    # Default to latest draft
    if args.ref is None:
        draft = all_drafts[-1]
    else:
        draft = resolve_draft_ref(args.ref, all_drafts)
        if draft is None:
            valid_iters = [d.iter for d in all_drafts]
            print(f"Draft not found for reference {args.ref}.")
            print(f"Valid iteration numbers: {valid_iters}")
            print(f"Or use negative offsets: -1 (latest) to -{len(all_drafts)} (oldest)")
            return 1

    # Accept by the draft's internal index (1-based sequential)
    try:
        offset = get_draft_offset(draft, all_drafts)
        accepted = session.accept_draft(draft.index)
        session.save()

        print(f"Accepted draft #{accepted.iter} ({offset}):")
        print("-" * 40)
        print(accepted.text)
        print("-" * 40)
        print("Exchange added to history. Ready for next message.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_drafts_archive(args, session: SessionV2) -> int:
    """Show archived drafts from past exchanges."""
    dialogue_pool = session.dialogue_pool

    # If no refs provided, list all exchanges
    if not args.refs:
        exchanges = dialogue_pool.list_exchanges()
        if not exchanges:
            print("No archived exchanges yet.")
            print("Drafts are archived when you accept a draft with 'mind accept'.")
            return 0

        print(f"Archived exchanges ({len(exchanges)} total):")
        print()
        for ex in exchanges:
            accepted = f"accepted #{ex['accepted_index']}" if ex['accepted_index'] else "no accepted"
            iter_range = f"iter {ex['first_iter']}-{ex['last_iter']}" if ex['first_iter'] != ex['last_iter'] else f"iter {ex['first_iter']}"
            print(f"  {ex['exchange_id']}: {ex['draft_count']} draft(s), {accepted}, {iter_range}")

        print()
        print("Usage: mind drafts archive <exchange_id>  # Show all drafts for that exchange")
        return 0

    # Show drafts for a specific exchange
    exchange_id = args.refs[0]
    drafts = dialogue_pool.get_exchange_drafts(exchange_id)

    if not drafts:
        print(f"No drafts found for exchange '{exchange_id}'.")
        exchanges = dialogue_pool.list_exchanges()
        if exchanges:
            print(f"Valid exchange IDs: {[e['exchange_id'] for e in exchanges]}")
        return 1

    print(f"Exchange {exchange_id} ({len(drafts)} draft(s)):")
    print()

    for draft in drafts:
        accepted = "[ACCEPTED]" if draft.get('accepted') else ""
        seen = "[seen]" if draft.get('user_seen') else "[unseen]"
        print(f"Draft #{draft['draft_index']} (iter {draft['iter_created']}) {seen} {accepted}")
        print("\u2500" * 20)

        # Truncate to max 8 lines
        truncated, remaining = truncate_text(draft['text'], max_lines=8)
        for line in truncated.split('\n'):
            print(f"  {line}")
        if remaining > 0:
            print(f"  ... ({remaining} more lines)")
        print()

    return 0


def cmd_drafts(args) -> int:
    """Show or manage drafts."""
    session_dir = get_current_session_dir()
    session = SessionV2(session_dir)

    # Handle 'archive' subcommand - works regardless of drafting state
    if args.drafts_command == 'archive':
        return cmd_drafts_archive(args, session)

    if not session.is_drafting:
        print("No pending message. Send a message with 'mind message' first.")
        return 0

    all_drafts = session.get_all_drafts()

    # Handle 'show' subcommand
    if args.drafts_command == 'show':
        if not args.refs:
            print("Usage: mind drafts show <iter|offset>")
            print("  mind drafts show 247    # Show draft from iteration 247")
            print("  mind drafts show -1     # Show latest draft")
            return 1

        if not all_drafts:
            print("No drafts available.")
            return 1

        ref = int(args.refs[0])
        draft = resolve_draft_ref(ref, all_drafts)
        if draft is None:
            valid_iters = [d.iter for d in all_drafts]
            print(f"Draft not found for reference {ref}.")
            print(f"Valid iteration numbers: {valid_iters}")
            print(f"Or use negative offsets: -1 (latest) to -{len(all_drafts)} (oldest)")
            return 1

        offset = get_draft_offset(draft, all_drafts)
        seen = "[seen]" if draft.seen else "[unseen]"
        print(f"Draft #{draft.iter} ({offset}) {seen}")
        print("-" * 40)
        print(draft.text)
        print("-" * 40)
        return 0

    # Handle 'seen' subcommand
    if args.drafts_command == 'seen':
        if not all_drafts:
            print("No drafts to mark as seen.")
            return 0

        if args.refs:
            # Mark specific drafts as seen (by iter or offset)
            marked = []
            for ref_str in args.refs:
                ref = int(ref_str)
                draft = resolve_draft_ref(ref, all_drafts)
                if draft:
                    draft.seen = True
                    marked.append(draft.iter)
                else:
                    print(f"Warning: draft not found for reference {ref}")
            if marked:
                session.dialogue_pool.save()
                print(f"Marked drafts as seen: {marked}")
        else:
            # Mark all as seen
            session.mark_drafts_seen(None)
            session.save()
            print("Marked all drafts as seen.")
        return 0

    # Default: show all drafts (truncated)
    awaiting = session.dialogue_pool.awaiting
    print("Awaiting response to:")
    print("-" * 40)
    # Truncate awaiting message too
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
        offset = get_draft_offset(draft, all_drafts)
        seen = "[seen]" if draft.seen else "[unseen]"
        print(f"Draft #{draft.iter} ({offset}) {seen}")
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
    print(f"  mind accept              Accept latest draft (#{latest.iter})")
    print(f"  mind accept <iter>       Accept draft by iteration number")
    print(f"  mind accept -N           Accept Nth-from-latest draft")
    print(f"  mind drafts show <ref>   Show full draft text")
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

    print(f"Conversation history ({len(history) // 2} exchanges):")
    print("=" * 60)

    current_exchange = 0
    for i, entry in enumerate(history):
        if entry.role == "user":
            current_exchange += 1
            print(f"\n[Exchange {current_exchange}]")
            print(f"User (age: {session.iteration - entry.iter}):")
        else:
            print(f"\nMind (age: {session.iteration - entry.iter}):")

        # Show text with indentation
        for line in entry.text.split('\n'):
            print(f"  {line}")

    print("\n" + "=" * 60)
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
    p_run = subparsers.add_parser('run', help='Run until draft (or N iterations)')
    p_run.add_argument('iterations', type=int, nargs='?', default=None,
                       help='Number of iterations (default: run until draft)')
    p_run.add_argument('--max', type=int, default=100,
                       help='Max iterations when running until draft (default: 100)')
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
    p_accept.add_argument('ref', type=int, nargs='?', default=None,
                          help='Draft reference: iter number or negative offset (-1=latest)')

    # drafts
    p_drafts = subparsers.add_parser('drafts', help='Show or manage drafts')
    p_drafts.add_argument('drafts_command', nargs='?', choices=['seen', 'show', 'archive'],
                          help='Subcommand: seen, show, archive')
    p_drafts.add_argument('refs', nargs='*', help='Draft references (iter or negative offset) or exchange_id for archive')

    # history
    subparsers.add_parser('history', help='Show conversation history')

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
        'accept': cmd_accept,
        'drafts': cmd_drafts,
        'history': cmd_history,
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
