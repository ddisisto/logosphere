"""
Mind v2 - YAML-based Mind invocation for Logosphere v2.

Implements the LOGOSPHERE MIND PROTOCOL v1.2:
- YAML input format (meta + thinking_pool + dialogue)
- Custom comment-based metadata for thoughts
- YAML output parsing (thoughts + draft)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from .thinking_pool import Thought
from .dialogue_pool import DialoguePool, Draft, HistoryEntry, UserMessage


# ============================================================================
# System Prompt (protocol spec)
# ============================================================================

def load_system_prompt() -> str:
    """Load system prompt from docs/system_prompt_v1.2.md."""
    prompt_path = Path(__file__).parent.parent.parent / 'docs' / 'system_prompt_v1.2.md'
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


def format_draft_yaml(draft: Draft, current_iter: int) -> str:
    """
    Format a single draft as YAML with comment metadata.

    Format:
      - |  # age: N, user_seen: true/false
        draft text here
    """
    age = current_iter - draft.iter
    seen_str = 'true' if draft.seen else 'false'

    # Indent text for YAML block
    lines = draft.text.split('\n')
    indented_text = '\n'.join('      ' + line for line in lines)

    return f'    - |  # age: {age}, user_seen: {seen_str}\n{indented_text}'


def format_history_entry_yaml(entry: HistoryEntry, current_iter: int) -> str:
    """
    Format a history entry as YAML.

    Format:
      - from: user/self
        age: N
        text: |
          message text
    """
    age = current_iter - entry.iter
    role = 'self' if entry.role == 'mind' else entry.role

    # Indent text for YAML block
    lines = entry.text.split('\n')
    indented_text = '\n'.join('        ' + line for line in lines)

    return f'''    - from: {role}
      age: {age}
      text: |
{indented_text}'''


def format_input(
    mind_id: str,
    current_iter: int,
    thoughts: list[Thought],
    dialogue_pool: DialoguePool,
    cluster_assignments: Optional[dict] = None,  # vector_id -> {cluster_id, size}
    user_time: Optional[str] = None,
) -> str:
    """
    Format input for Mind invocation (v1.2 protocol).

    Args:
        mind_id: Identity of this mind (e.g., "mind_0")
        current_iter: Current iteration number
        thoughts: Sampled thoughts from thinking pool
        dialogue_pool: Dialogue pool with awaiting/drafts/history
        cluster_assignments: Optional cluster info per thought {cluster_id, size}
        user_time: Optional timestamp override

    Returns:
        YAML string for Mind input with dialogue section
    """
    if user_time is None:
        user_time = datetime.now(timezone.utc).isoformat()

    # Build meta section
    meta_yaml = f'''meta:
  self: {mind_id}
  iter: {current_iter}
  user_time: {user_time}'''

    # Build thinking pool with custom comment format
    thought_items = []
    for thought in thoughts:
        cluster_info = None
        if cluster_assignments and thought.vector_id is not None:
            cluster_info = cluster_assignments.get(thought.vector_id)
        thought_items.append(format_thought_yaml(thought, cluster_info, current_iter))

    thinking_yaml = 'thinking_pool:\n  # A *random, unordered sample* from the pool. What should be remembered?'
    if thought_items:
        thinking_yaml += '\n' + '\n'.join(thought_items)

    # Build dialogue section
    history = dialogue_pool.get_history()

    if dialogue_pool.is_drafting:
        # Drafting state: show history (if any) + awaiting message + drafts
        dialogue_yaml = 'dialogue:'

        # Include history for context
        if history:
            history_items = [format_history_entry_yaml(h, current_iter) for h in history]
            dialogue_yaml += '\n  # Conversation history for context\n  history:\n' + '\n'.join(history_items)

        # Show awaiting message
        awaiting = dialogue_pool.awaiting
        awaiting_age = current_iter - awaiting.iter
        awaiting_lines = awaiting.text.split('\n')
        awaiting_indented = '\n'.join('      ' + line for line in awaiting_lines)

        dialogue_yaml += f'''

  # User's message awaiting your response
  awaiting:
    age: {awaiting_age}
    text: |
{awaiting_indented}'''

        # Add drafts if any
        if dialogue_pool.drafts:
            draft_items = [format_draft_yaml(d, current_iter) for d in dialogue_pool.drafts]
            dialogue_yaml += '''

  # Your draft responses (most recent = last in list)
  drafts:
''' + '\n'.join(draft_items)

    else:
        # Idle state: show history only
        dialogue_yaml = 'dialogue:\n  # No pending user message. Conversation history for context.'
        if history:
            history_items = [format_history_entry_yaml(h, current_iter) for h in history]
            dialogue_yaml += '\n  history:\n' + '\n'.join(history_items)

    return f'{meta_yaml}\n\n{thinking_yaml}\n\n{dialogue_yaml}\n'


# ============================================================================
# Output Parsing
# ============================================================================

class MindOutput:
    """Parsed Mind output."""

    def __init__(
        self,
        thoughts: list[str],
        draft: Optional[str] = None,
        skipped: bool = False,
        raw: str = '',
    ):
        self.thoughts = thoughts
        self.draft = draft
        self.skipped = skipped
        self.raw = raw


def parse_output(raw: str) -> MindOutput:
    """
    Parse Mind YAML output.

    Handles:
    - thoughts: list of strings
    - draft: single string (optional)
    - skip: true (explicit opt-out)
    - Empty arrays (valid)

    Returns:
        MindOutput with parsed thoughts and draft
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
        return MindOutput(thoughts=[], raw=raw)

    if data is None:
        return MindOutput(thoughts=[], raw=raw)

    # Check for explicit skip
    if data.get('skip') is True:
        return MindOutput(thoughts=[], skipped=True, raw=raw)

    # Extract thoughts
    thoughts = data.get('thoughts', [])
    if thoughts is None:
        thoughts = []
    # Ensure all thoughts are strings
    thoughts = [str(t).strip() for t in thoughts if t]

    # Extract draft
    draft = data.get('draft')
    if draft is not None:
        draft = str(draft).strip()
        if not draft:
            draft = None

    return MindOutput(
        thoughts=thoughts,
        draft=draft,
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
        system_prompt: Protocol spec (from docs/system_prompt_v1.2.md)
        user_input: Formatted YAML input
        model: Model to use
        token_limit: Max tokens for response
        api_key: Optional API key override

    Returns:
        MindOutput with parsed thoughts and draft
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
