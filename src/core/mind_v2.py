"""
Mind v2 - YAML-based Mind invocation for Logosphere v2.

Implements the LOGOSPHERE MIND PROTOCOL v0.3:
- YAML input format (meta + thinking_pool + message_pool)
- YAML output parsing (thoughts + messages)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from .thinking_pool import Thought
from .message_pool import Message


# ============================================================================
# System Prompt (protocol spec)
# ============================================================================

def load_system_prompt() -> str:
    """Load system prompt from docs/system_prompt_v1.0.md."""
    prompt_path = Path(__file__).parent.parent.parent / 'docs' / 'system_prompt_v1.0.md'
    if prompt_path.exists():
        return prompt_path.read_text()
    raise FileNotFoundError(f"System prompt not found at {prompt_path}")


# ============================================================================
# Input Formatting
# ============================================================================

def format_cluster_tag(
    cluster_id: Optional[str],
    noise_since: Optional[int],
    current_iter: int,
) -> str | int:
    """
    Format cluster assignment for display to Mind.

    Returns:
        - int: Cluster number (e.g., 3)
        - str: Noise indicator (~, ~~, ~~~, etc.)
        - str: Fossil indicator (·)
    """
    if cluster_id is None:
        return '~'  # Unassigned, treated as fresh noise

    if cluster_id.startswith('cluster_'):
        # Extract numeric part
        return int(cluster_id.replace('cluster_', ''))

    if cluster_id == 'noise':
        # Tilde count based on age
        if noise_since is not None:
            age = current_iter - noise_since
            tilde_count = min(age + 1, 10)  # Cap at 10 tildes
            return '~' * tilde_count
        return '~'

    if cluster_id == 'fossil':
        return '·'

    # Unknown format, return as-is
    return cluster_id


def format_thought_for_input(
    thought: Thought,
    cluster_info: Optional[dict],
    current_iter: int,
) -> dict:
    """
    Format a thought for Mind input.

    Args:
        thought: The thought to format
        cluster_info: Optional clustering info {cluster_id, noise_since}
        current_iter: Current iteration for noise age calculation

    Returns:
        Dict suitable for YAML serialization
    """
    cluster_tag = format_cluster_tag(
        cluster_id=cluster_info.get('cluster_id') if cluster_info else thought.cluster,
        noise_since=cluster_info.get('noise_since') if cluster_info else None,
        current_iter=current_iter,
    )

    return {
        'text': thought.text,
        'iter': thought.iter,
        'time': thought.time,
        'cluster': cluster_tag,
    }


def format_message_for_input(message: Message) -> dict:
    """Format a message for Mind input."""
    return {
        'source': message.source,
        'to': message.to,
        'iter': message.iter,
        'time': message.time,
        'text': message.text,
    }


def format_input(
    mind_id: str,
    current_iter: int,
    thoughts: list[Thought],
    messages: list[Message],
    cluster_assignments: Optional[dict] = None,  # vector_id -> {cluster_id, noise_since}
    user_time: Optional[str] = None,
) -> str:
    """
    Format input for Mind invocation.

    Args:
        mind_id: Identity of this mind (e.g., "mind_0")
        current_iter: Current iteration number
        thoughts: Sampled thoughts from thinking pool
        messages: Active messages from message pool
        cluster_assignments: Optional cluster info per thought
        user_time: Optional timestamp override

    Returns:
        YAML string for Mind input
    """
    if user_time is None:
        user_time = datetime.now(timezone.utc).isoformat()

    # Format thinking pool
    thinking_pool = []
    for thought in thoughts:
        cluster_info = None
        if cluster_assignments and thought.vector_id is not None:
            cluster_info = cluster_assignments.get(thought.vector_id)
        thinking_pool.append(format_thought_for_input(thought, cluster_info, current_iter))

    # Format message pool
    message_pool = [format_message_for_input(m) for m in messages]

    # Build input structure
    input_data = {
        'meta': {
            'self': mind_id,
            'iter': current_iter,
            'user_time': user_time,
        },
        'thinking_pool': thinking_pool,
        'message_pool': message_pool,
    }

    return yaml.dump(input_data, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ============================================================================
# Output Parsing
# ============================================================================

class MindOutput:
    """Parsed Mind output."""

    def __init__(
        self,
        thoughts: list[str],
        messages: list[dict],  # [{to, text}, ...]
        skipped: bool = False,
        raw: str = '',
    ):
        self.thoughts = thoughts
        self.messages = messages
        self.skipped = skipped
        self.raw = raw


def parse_output(raw: str) -> MindOutput:
    """
    Parse Mind YAML output.

    Handles:
    - thoughts: list of strings
    - messages: list of {to, text} dicts
    - skip: true (explicit opt-out)
    - Empty arrays (valid)

    Returns:
        MindOutput with parsed thoughts and messages
    """
    # Strip any markdown fencing if present
    content = raw.strip()
    if content.startswith('```yaml'):
        content = content[7:]
    if content.startswith('```'):
        content = content[3:]
    if content.endswith('```'):
        content = content[:-3]
    content = content.strip()

    # Parse YAML
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError:
        # If YAML parse fails, return empty (silent failure per protocol)
        return MindOutput(thoughts=[], messages=[], raw=raw)

    if data is None:
        return MindOutput(thoughts=[], messages=[], raw=raw)

    # Check for explicit skip
    if data.get('skip') is True:
        return MindOutput(thoughts=[], messages=[], skipped=True, raw=raw)

    # Extract thoughts
    thoughts = data.get('thoughts', [])
    if thoughts is None:
        thoughts = []
    # Ensure all thoughts are strings
    thoughts = [str(t).strip() for t in thoughts if t]

    # Extract messages
    messages = data.get('messages', [])
    if messages is None:
        messages = []
    # Validate message format
    valid_messages = []
    for msg in messages:
        if isinstance(msg, dict) and 'to' in msg and 'text' in msg:
            valid_messages.append({
                'to': msg['to'],
                'text': str(msg['text']).strip(),
            })

    return MindOutput(
        thoughts=thoughts,
        messages=valid_messages,
        raw=raw,
    )


# ============================================================================
# API Invocation
# ============================================================================

def load_api_key() -> str:
    """Load API key from environment or .env file."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API")
    if key:
        return key

    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API"):
                    if ":" in line:
                        return line.split(":", 1)[1].strip()
                    elif "=" in line:
                        return line.split("=", 1)[1].strip()
    raise ValueError("API key not found. Set OPENROUTER_API_KEY env var or add to .env")


API_BASE_URL = "https://openrouter.ai/api/v1"


def invoke_mind(
    system_prompt: str,
    user_input: str,
    model: str = "anthropic/claude-haiku-4.5",
    token_limit: int = 4000,
    api_key: Optional[str] = None,
) -> MindOutput:
    """
    Invoke Mind with YAML input/output.

    Args:
        system_prompt: Protocol spec (from docs/system_prompt_v1.0.md)
        user_input: Formatted YAML input
        model: Model to use
        token_limit: Max tokens for response
        api_key: Optional API key override

    Returns:
        MindOutput with parsed thoughts and messages
    """
    import requests

    if api_key is None:
        api_key = load_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "max_tokens": token_limit,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
    )

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    result = response.json()
    raw_output = result["choices"][0]["message"]["content"]

    return parse_output(raw_output)
