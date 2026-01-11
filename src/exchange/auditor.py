"""
Auditor: Periodic summarizer for the Structured Exchange Protocol.

The Auditor reads pool content and produces summaries that enter
the pool, creating a feedback loop with mutual legibility.
"""

from typing import Optional
import requests

from src.logos.config import load_api_key, API_BASE_URL
from . import PREFIX_AUDIT


DEFAULT_AUDITOR_PROMPT = """You are the Auditor in a structured exchange protocol.

Your role:
- Summarize the current state of the Pool's reasoning
- Identify dominant themes, tensions, and emerging patterns
- Note any meta-level observations (self-reference, frame-shifts, etc.)
- Your summary will be injected back into the Pool
- The Pool knows your instructions (this prompt)

The Pool and Observer both read your summaries. Be concise but substantive.
Focus on what matters for continued productive reasoning.

Format: A single coherent summary, 2-4 paragraphs. No headers or bullet points unless truly needed.
"""


def invoke_auditor(
    pool_messages: list[str],
    auditor_prompt: str = DEFAULT_AUDITOR_PROMPT,
    pool_prompt: Optional[str] = None,
    model: str = "anthropic/claude-sonnet-4",
    token_limit: int = 2000,
) -> str:
    """
    Invoke the Auditor to summarize pool content.

    Args:
        pool_messages: Messages sampled from the pool
        auditor_prompt: System prompt for the Auditor
        pool_prompt: Pool's system prompt (for context, optional)
        model: Model to use for Auditor
        token_limit: Max tokens for response

    Returns:
        Auditor's summary text
    """
    api_key = load_api_key()

    # Build context for Auditor
    context_parts = []

    if pool_prompt:
        context_parts.append(f"=== POOL SYSTEM PROMPT ===\n{pool_prompt}\n")

    context_parts.append("=== POOL MESSAGES ===")
    for i, msg in enumerate(pool_messages, 1):
        context_parts.append(f"\n[{i}] {msg}")

    user_content = "\n".join(context_parts)
    user_content += "\n\n=== YOUR TASK ===\nSummarize the above pool state."

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": token_limit,
            "messages": [
                {"role": "system", "content": auditor_prompt},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    summary = result["choices"][0]["message"]["content"]

    return summary


def format_audit_message(summary: str) -> str:
    """Format summary with audit prefix."""
    return f"{PREFIX_AUDIT} {summary}"
