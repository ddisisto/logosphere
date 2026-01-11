"""
Auditor hook for Structured Exchange Protocol.

Every N rounds, invokes the Auditor to summarize pool state.
Summary is injected back into pool with [AUDIT] prefix.
"""

from pathlib import Path

# Configuration
AUDIT_EVERY = 20  # Audit every N iterations
AUDITOR_MODEL = "anthropic/claude-sonnet-4.5"
SAMPLE_SIZE = 30  # Messages to show auditor (or None for all active)

# Auditor prompt - can be overridden via config
AUDITOR_PROMPT = """You are the Auditor in a structured exchange protocol.

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


def hook(session, iteration: int, runner) -> None:
    """
    Called before each iteration.

    Every AUDIT_EVERY iterations, invokes Auditor and injects summary.
    """
    if iteration == 0:
        return

    if iteration % AUDIT_EVERY != 0:
        return

    print(f"[auditor] Running audit at iteration {iteration}...")

    # Import here to avoid circular deps
    from src.exchange.auditor import invoke_auditor, format_audit_message

    # Get pool messages for auditor
    # Sample broadly - more than normal k_samples
    sample_size = SAMPLE_SIZE or runner.config.active_pool_size
    messages, _ = session.sample(sample_size)

    if not messages:
        print("[auditor] No messages to audit")
        return

    # Get pool system prompt for context
    pool_prompt = runner.config.system_prompt

    # Check for custom auditor prompt in config
    auditor_prompt = session.config.get('auditor_prompt', AUDITOR_PROMPT)
    auditor_model = session.config.get('auditor_model', AUDITOR_MODEL)

    # Invoke auditor
    summary = invoke_auditor(
        pool_messages=messages,
        auditor_prompt=auditor_prompt,
        pool_prompt=pool_prompt,
        model=auditor_model,
    )

    # Inject summary into pool
    audit_message = format_audit_message(summary)
    session.inject_message(
        text=audit_message,
        embedding_client=runner.embedding_client,
        notes=f"auditor hook @ iteration {iteration}",
    )

    # Preview
    preview = summary[:100] + "..." if len(summary) > 100 else summary
    print(f"[auditor] Injected: {preview}")
