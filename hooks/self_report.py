"""
Self-report injection hook for Structured Exchange Protocol.

Injects role descriptions so all parties have mutual legibility:
- [OBSERVER ROLE] - Human's stated role and powers
- [AUDITOR ROLE] - Auditor's system prompt

Typically runs at same cadence as auditor, or at session start.
"""

from pathlib import Path

# Configuration
REPORT_EVERY = 20  # Same cadence as auditor
REPORT_AT_START = True  # Also inject at iteration 0

# Default paths - can be overridden in config
OBSERVER_ROLE_FILE = Path(__file__).parent.parent / ".daniel" / "ROLE.md"

# Auditor prompt (same as in auditor.py - keep in sync or read from config)
DEFAULT_AUDITOR_PROMPT = """You are the Auditor in a structured exchange protocol.

Your role:
- Summarize the current state of the Pool's reasoning
- Identify dominant themes, tensions, and emerging patterns
- Note any meta-level observations (self-reference, frame-shifts, requests)
- Your summary will be injected back into the Pool
- The Pool knows your instructions (this prompt is visible to them)

The Pool and Observer both read your summaries. Be concise but substantive.
Focus on what matters for continued productive reasoning.

Format: A single coherent summary, 2-4 paragraphs.
"""


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

    Injects role descriptions at start and every REPORT_EVERY iterations.
    """
    from src.exchange import PREFIX_OBSERVER_ROLE, PREFIX_AUDITOR_ROLE

    # Decide whether to inject this iteration
    should_inject = False
    if REPORT_AT_START and iteration == 0:
        should_inject = True
    elif iteration > 0 and iteration % REPORT_EVERY == 0:
        should_inject = True

    if not should_inject:
        return

    print(f"[self_report] Injecting role descriptions at iteration {iteration}...")

    # Get auditor prompt from config or default
    auditor_prompt = session.config.get('auditor_prompt', DEFAULT_AUDITOR_PROMPT)

    # Get observer role
    observer_role = load_observer_role(session.config)

    # Inject auditor role
    auditor_message = f"{PREFIX_AUDITOR_ROLE} {auditor_prompt}"
    session.inject_message(
        text=auditor_message,
        embedding_client=runner.embedding_client,
        notes=f"self_report hook (auditor) @ iteration {iteration}",
    )
    print(f"[self_report] Injected auditor role ({len(auditor_prompt)} chars)")

    # Inject observer role
    observer_message = f"{PREFIX_OBSERVER_ROLE} {observer_role}"
    session.inject_message(
        text=observer_message,
        embedding_client=runner.embedding_client,
        notes=f"self_report hook (observer) @ iteration {iteration}",
    )
    print(f"[self_report] Injected observer role ({len(observer_role)} chars)")
