"""
Mind v2 - YAML-based Mind invocation for Logosphere v2.

Implements the LOGOSPHERE MIND PROTOCOL v1.1:
- YAML input format (meta + thinking_pool + message_pool)
- Custom comment-based metadata for thoughts
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
    """Load system prompt from docs/system_prompt_v1.1.md."""
    prompt_path = Path(__file__).parent.parent.parent / 'docs' / 'system_prompt_v1.1.md'
    if prompt_path.exists():
        return prompt_path.read_text()
    raise FileNotFoundError(f"System prompt not found at {prompt_path}")


# ============================================================================
# Input Formatting
# ============================================================================

def format_cluster_comment(cluster_id: Optional[str], cluster_size: Optional[int]) -> str:
    """
    Format cluster info for YAML comment.

    Returns:
        - '{id: N, size: M}' for clustered thoughts
        - '{~}' for noise/unassigned
    """
    if cluster_id is None or cluster_id == 'noise':
        return '{~}'

    if cluster_id.startswith('cluster_'):
        cid = int(cluster_id.replace('cluster_', ''))
        size = cluster_size if cluster_size is not None else 1
        return f'{{id: {cid}, size: {size}}}'

    # Unknown format
    return '{~}'


def format_thought_yaml(
    thought: Thought,
    cluster_info: Optional[dict],
    current_iter: int,
) -> str:
    """
    Format a single thought as YAML with comment metadata.

    Format:
      - |  # age: N, cluster: {id: M, size: K}
        thought text here
        possibly multiple lines
    """
    age = current_iter - thought.iter
    cluster_id = cluster_info.get('cluster_id') if cluster_info else thought.cluster
    cluster_size = cluster_info.get('size') if cluster_info else None
    cluster_comment = format_cluster_comment(cluster_id, cluster_size)

    # Indent text for YAML block
    lines = thought.text.split('\n')
    indented_text = '\n'.join('    ' + line for line in lines)

    return f'  - |  # age: {age}, cluster: {cluster_comment}\n{indented_text}'


def format_message_yaml(message: Message, current_iter: int) -> str:
    """
    Format a single message as YAML.

    Format:
      - source: user
        to: mind_0
        age: 162
        time: 2026-01-15T12:31:19+11:00
        text: |
          message text here
    """
    age = current_iter - message.iter

    # Indent text for YAML block
    lines = message.text.split('\n')
    indented_text = '\n'.join('      ' + line for line in lines)

    return f'''  - source: {message.source}
    to: {message.to}
    age: {age}
    time: {message.time}
    text: |
{indented_text}'''


def format_input(
    mind_id: str,
    current_iter: int,
    thoughts: list[Thought],
    messages: list[Message],
    cluster_assignments: Optional[dict] = None,  # vector_id -> {cluster_id, size}
    user_time: Optional[str] = None,
) -> str:
    """
    Format input for Mind invocation (v1.1 protocol).

    Args:
        mind_id: Identity of this mind (e.g., "mind_0")
        current_iter: Current iteration number
        thoughts: Sampled thoughts from thinking pool
        messages: Active messages from message pool
        cluster_assignments: Optional cluster info per thought {cluster_id, size}
        user_time: Optional timestamp override

    Returns:
        YAML string for Mind input with comment-based thought metadata
    """
    if user_time is None:
        user_time = datetime.now(timezone.utc).isoformat()

    # Build meta section
    meta_yaml = f'''meta:
  self: {mind_id}
  # The clock of iterations marches ever forward. What persists and what changes?
  iter: {current_iter}
  user_time: {user_time}'''

    # Build thinking pool with custom comment format
    thought_items = []
    for thought in thoughts:
        cluster_info = None
        if cluster_assignments and thought.vector_id is not None:
            cluster_info = cluster_assignments.get(thought.vector_id)
        thought_items.append(format_thought_yaml(thought, cluster_info, current_iter))

    thinking_yaml = 'thinking_pool:\n  # A *random, unordered sample* from the pool. What should be remembered? What should be forgotten?'
    if thought_items:
        thinking_yaml += '\n' + '\n'.join(thought_items)

    # Build message pool
    message_items = [format_message_yaml(m, current_iter) for m in messages]
    messages_yaml = 'message_pool:\n  # Direct dialogue across the boundary. What deserves a response?'
    if message_items:
        messages_yaml += '\n' + '\n'.join(message_items)

    return f'{meta_yaml}\n\n{thinking_yaml}\n\n{messages_yaml}\n'


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
