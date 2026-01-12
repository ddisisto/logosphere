"""
Observer role injection hook for Structured Exchange Protocol.

Injects the Observer's role description so Pool has mutual legibility.
Runs at same cadence as auditor (after auditor hook).
"""

from pathlib import Path

# Configuration
INJECT_EVERY = 20  # Same cadence as auditor
INJECT_AT_START = True  # Also inject at iteration 0

# Default path - can be overridden in config
OBSERVER_ROLE_FILE = Path(__file__).parent.parent / ".daniel" / "ROLE.md"


def load_observer_role(config: dict) -> str:
    """Load observer role from file or config."""
    # Check config first
    if 'observer_role' in config:
        return config['observer_role']

    # Check config for custom path
    role_path = config.get('observer_role_file', OBSERVER_ROLE_FILE)
    if isinstance(role_path, str):
        role_path = Path(role_path)

    if role_path.exists():
        return role_path.read_text().strip()

    return "(Observer role not specified)"


def hook(session, iteration: int, runner) -> None:
    """
    Called before each iteration.

    Injects observer role at start and every INJECT_EVERY iterations.
    """
    from src.exchange import PREFIX_OBSERVER_ROLE

    # Decide whether to inject this iteration
    should_inject = False
    if INJECT_AT_START and iteration == 0:
        should_inject = True
    elif iteration > 0 and iteration % INJECT_EVERY == 0:
        should_inject = True

    if not should_inject:
        return

    print(f"[observer] Injecting observer role at iteration {iteration}...")

    # Get observer role
    observer_role = load_observer_role(session.config)

    # Inject observer role
    observer_message = f"{PREFIX_OBSERVER_ROLE} {observer_role}"
    session.inject_message(
        text=observer_message,
        embedding_client=runner.embedding_client,
        notes=f"observer hook @ iteration {iteration}",
    )
    print(f"[observer] Injected observer role ({len(observer_role)} chars)")
