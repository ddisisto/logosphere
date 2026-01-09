#!/usr/bin/env python3
"""
Logos CLI - Pool-based reasoning with branch-based history.

Usage:
    python scripts/logos.py init ./session "initial prompt"
    python scripts/logos.py run 10
    python scripts/logos.py step
    python scripts/logos.py inject "thought text"
    python scripts/logos.py fork experiment
    python scripts/logos.py switch main
    python scripts/logos.py status
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.session import Session
from src.core.embedding_client import EmbeddingClient
from src.logos.config import LogosConfig
from src.logos.runner import LogosRunner


# Session directory tracking (for commands that need an open session)
SESSION_FILE = Path.home() / ".logos_session"


def get_current_session_dir() -> Path:
    """Get the current session directory."""
    if SESSION_FILE.exists():
        return Path(SESSION_FILE.read_text().strip())
    raise RuntimeError("No session open. Use 'logos init' or 'logos open' first.")


def set_current_session_dir(session_dir: Path) -> None:
    """Set the current session directory."""
    SESSION_FILE.write_text(str(session_dir.absolute()))


def cmd_init(args):
    """Initialize a new session."""
    session_dir = Path(args.session_dir)

    if session_dir.exists() and (session_dir / "branches.json").exists():
        print(f"Session already exists at {session_dir}")
        print("Use 'logos open' to open it, or choose a different directory.")
        return 1

    # Parse config overrides
    config_dict = {}
    if args.model:
        config_dict['model'] = args.model
    if args.k_samples:
        config_dict['k_samples'] = args.k_samples
    if args.active_pool_size:
        config_dict['active_pool_size'] = args.active_pool_size

    config = LogosConfig(**config_dict)

    # Create session
    session = Session(
        session_dir=session_dir,
        active_pool_size=config.active_pool_size,
        embedding_dim=config.embedding_dim,
    )
    session.config = config.to_dict()

    # Seed with prompts
    if args.prompts:
        runner = LogosRunner(session, config)
        runner.seed_prompts(args.prompts)
        session._save()

    set_current_session_dir(session_dir)

    print(f"Session initialized: {session_dir}")
    print(f"Branch: {session.current_branch}")
    visible = len(session.get_visible_ids())
    print(f"Pool: {visible} messages")
    if args.prompts:
        print(f"Seeded with {len(args.prompts)} prompt(s)")

    return 0


def cmd_open(args):
    """Open an existing session."""
    session_dir = Path(args.session_dir)

    if not session_dir.exists():
        print(f"Session not found: {session_dir}")
        return 1

    if not (session_dir / "branches.json").exists():
        print(f"Not a valid session directory: {session_dir}")
        return 1

    set_current_session_dir(session_dir)
    print(f"Opened session: {session_dir}")

    # Show status
    session = Session(session_dir)
    status = session.get_status()
    print(f"Branch: {status['current_branch']}")
    print(f"Iteration: {status['iteration']}")
    print(f"Pool: {status['visible_messages']} visible ({status['total_messages']} total)")

    return 0


def cmd_run(args):
    """Run batch iterations."""
    session_dir = get_current_session_dir()
    session = Session(session_dir)

    # Load config
    config = LogosConfig.from_dict(session.config) if session.config else LogosConfig()

    # Apply overrides
    if args.converge:
        config.convergence_threshold = 0.75
        config.convergence_coverage = 0.50
    if not args.verbose:
        config.verbose = False

    runner = LogosRunner(session, config)

    # Inject additional prompts if provided
    if args.prompt:
        runner.seed_prompts([args.prompt])

    print(f"Running {args.iterations} iterations on branch '{session.current_branch}'...")
    print("-" * 40)

    metrics_history = runner.run(args.iterations)

    print("-" * 40)
    print(f"Completed {len(metrics_history)} iterations")
    if metrics_history:
        final = metrics_history[-1]
        print(f"Final: clusters={final.num_clusters}, coherence={final.coherence:.2f}")

    return 0


def cmd_step(args):
    """Run single iteration."""
    session_dir = get_current_session_dir()
    session = Session(session_dir)

    config = LogosConfig.from_dict(session.config) if session.config else LogosConfig()
    runner = LogosRunner(session, config)

    print(f"Running step on branch '{session.current_branch}'...")
    metrics = runner.step()

    # Save after step
    session._save()

    print(f"Iteration {metrics.iteration}: {metrics.thoughts_added} thoughts added")
    print(f"Clusters: {metrics.num_clusters}, Coherence: {metrics.coherence:.2f}")

    return 0


def cmd_inject(args):
    """Inject a thought into the pool."""
    session_dir = get_current_session_dir()
    session = Session(session_dir)

    config = LogosConfig.from_dict(session.config) if session.config else LogosConfig()
    embedding_client = EmbeddingClient(model=config.embedding_model, enabled=True)

    intervention = session.inject_message(
        text=args.text,
        embedding_client=embedding_client,
        notes=args.notes or "",
    )

    print(f"Injected: {args.text[:60]}...")
    print(f"Branch: {session.current_branch}")

    return 0


def cmd_fork(args):
    """Create a new branch from current state."""
    session_dir = get_current_session_dir()
    session = Session(session_dir)

    old_branch = session.current_branch
    new_branch = session.fork(args.name)

    print(f"Forked '{old_branch}' -> '{new_branch}' at iteration {session.iteration}")
    print(f"Now on branch: {new_branch}")

    return 0


def cmd_switch(args):
    """Switch to a different branch."""
    session_dir = get_current_session_dir()
    session = Session(session_dir)

    old_branch = session.current_branch
    session.switch(args.name)

    print(f"Switched: {old_branch} -> {args.name}")
    visible = len(session.get_visible_ids())
    print(f"Visible pool: {visible} messages")

    return 0


def cmd_status(args):
    """Show session status."""
    session_dir = get_current_session_dir()
    session = Session(session_dir)

    status = session.get_status()

    print(f"Session: {status['session_dir']}")
    print(f"Branch: {status['current_branch']}")
    print(f"Iteration: {status['iteration']}")
    print(f"Pool: {status['visible_messages']} visible, {status['active_pool_size']} active")
    print(f"Total messages: {status['total_messages']}")
    print(f"Branches: {', '.join(status['branches'])}")
    print(f"Interventions: {status['interventions']}")

    return 0


def cmd_branches(args):
    """List all branches."""
    session_dir = get_current_session_dir()
    session = Session(session_dir)

    branches = session.list_branches()

    if not branches:
        print("No branches.")
        return 0

    for b in branches:
        marker = " *" if b['current'] else ""
        print(f"  {b['name']}{marker}")
        if b['fork_from']:
            print(f"    forked from: {b['fork_from']} @ iteration {b['fork_iteration']}")
        print(f"    own messages: {b['own_messages']}")
        print()

    return 0


def cmd_log(args):
    """Show intervention log."""
    session_dir = get_current_session_dir()
    session = Session(session_dir)

    interventions = session.intervention_log.query(limit=args.limit)

    if not interventions:
        print("No interventions recorded.")
        return 0

    for i in interventions:
        ts = i.timestamp[:19].replace('T', ' ')
        print(f"[{ts}] {i.intervention_type}: {i.id}")
        if i.notes:
            print(f"    notes: {i.notes}")
        print(f"    {i.snapshot_before} -> {i.snapshot_after}")
        print()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Logos - Pool-based reasoning with branch-based history"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init
    p_init = subparsers.add_parser("init", help="Initialize new session")
    p_init.add_argument("session_dir", help="Session directory")
    p_init.add_argument("prompts", nargs="*", help="Initial prompts to seed")
    p_init.add_argument("--model", help="LLM model")
    p_init.add_argument("--k-samples", type=int, help="Samples per iteration")
    p_init.add_argument("--active-pool-size", type=int, help="Active pool size")

    # open
    p_open = subparsers.add_parser("open", help="Open existing session")
    p_open.add_argument("session_dir", help="Session directory")

    # run
    p_run = subparsers.add_parser("run", help="Run batch iterations")
    p_run.add_argument("iterations", type=int, help="Number of iterations")
    p_run.add_argument("--prompt", help="Additional prompt to inject")
    p_run.add_argument("--converge", action="store_true",
                       help="Enable convergence-based early termination")
    p_run.add_argument("--verbose", action="store_true", default=True)
    p_run.add_argument("--quiet", dest="verbose", action="store_false")

    # step
    p_step = subparsers.add_parser("step", help="Run single iteration")

    # inject
    p_inject = subparsers.add_parser("inject", help="Inject thought into pool")
    p_inject.add_argument("text", help="Thought text")
    p_inject.add_argument("--notes", help="Observer notes")

    # fork
    p_fork = subparsers.add_parser("fork", help="Create new branch from current state")
    p_fork.add_argument("name", help="New branch name")

    # switch
    p_switch = subparsers.add_parser("switch", help="Switch to existing branch")
    p_switch.add_argument("name", help="Branch name")

    # status
    p_status = subparsers.add_parser("status", help="Show session status")

    # branches
    p_branches = subparsers.add_parser("branches", help="List all branches")

    # log
    p_log = subparsers.add_parser("log", help="Show intervention log")
    p_log.add_argument("--limit", type=int, default=20, help="Max entries")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch
    commands = {
        "init": cmd_init,
        "open": cmd_open,
        "run": cmd_run,
        "step": cmd_step,
        "inject": cmd_inject,
        "fork": cmd_fork,
        "switch": cmd_switch,
        "status": cmd_status,
        "branches": cmd_branches,
        "log": cmd_log,
    }

    try:
        return commands[args.command](args)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
