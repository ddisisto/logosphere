"""
Mind invocation and message parsing for Logosphere experiment.

Each Mind is a single stateless API call.
"""

import requests
from .. import config
from ..config import API_KEY, MODEL, API_BASE_URL, DELIMITER, TOKEN_LIMIT


def format_input(system_prompt: str, messages: list[str]) -> str:
    """
    Format input for Mind invocation.

    Structure:
        system_prompt
        ---
        message_1
        ---
        message_2
        ---
        ...
    """
    if not messages:
        return system_prompt

    formatted_messages = f"\n{DELIMITER}\n".join(messages)
    return f"{system_prompt}\n{DELIMITER}\n{formatted_messages}"


def parse_output(raw: str) -> tuple[str, list[str]]:
    """
    Parse Mind output into components.

    Rules:
    - Split on DELIMITER (---) on its own line
    - First block = thinking (private, not transmitted)
    - All subsequent blocks = messages (transmitted)
    - Blank blocks are valid messages (transmitted as-is)
    - No termination requirement - partial output is valid

    Returns:
        (thinking, messages)

    where:
        thinking: str - private reasoning before first ---
        messages: list[str] - all messages to transmit
    """
    # Normalize: ensure trailing newline for consistent split behavior
    if not raw.endswith('\n'):
        raw = raw + '\n'

    # Handle case where output starts with delimiter (no thinking)
    if raw.startswith(f"{DELIMITER}\n"):
        raw = "\n" + raw  # Add leading newline so split pattern matches

    # Split on delimiter
    delimiter_pattern = f"\n{DELIMITER}\n"
    blocks = raw.split(delimiter_pattern)

    # First block is thinking
    thinking = blocks[0].lstrip('\n') if blocks else ""

    # All remaining blocks are messages (including blank ones)
    messages = blocks[1:] if len(blocks) > 1 else []

    return thinking, messages


def invoke_mind(system_prompt: str, messages: list[str], token_limit: int = TOKEN_LIMIT) -> dict:
    """
    Invoke LLM Mind with formatted input.

    Args:
        system_prompt: System-level framing
        messages: Sampled messages from pool
        token_limit: Maximum tokens for generation

    Returns:
        {
            'thinking': str,
            'transmitted': list[str],
            'tokens_used': int,
            'raw_output': str
        }

    Raises:
        Fails fast on API errors (no catching)
    """
    # Format input
    formatted_input = format_input(system_prompt, messages)

    # Prepare API request
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": formatted_input}
        ],
        "max_tokens": token_limit
    }

    # Call API (fail fast on errors)
    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()

    # Extract response
    data = response.json()
    raw_output = data['choices'][0]['message']['content']
    tokens_used = data['usage']['completion_tokens']

    # Parse output
    thinking, transmitted = parse_output(raw_output)

    return {
        'thinking': thinking,
        'transmitted': transmitted,
        'tokens_used': tokens_used,
        'raw_output': raw_output
    }
