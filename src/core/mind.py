"""
Mind invocation and message parsing for Logosphere experiment.

Each Mind is a single stateless API call.
"""

import os
import requests
from pathlib import Path

# Message delimiter
DELIMITER = "---"

# API defaults
API_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
DEFAULT_TOKEN_LIMIT = 4000


def _load_api_key() -> str:
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


def invoke_mind(
    system_prompt: str,
    messages: list[str],
    token_limit: int = DEFAULT_TOKEN_LIMIT,
    model: str = None,
    api_key: str = None,
) -> dict:
    """
    Invoke LLM Mind with formatted input.

    Args:
        system_prompt: System-level framing
        messages: Sampled messages from pool
        token_limit: Maximum tokens for generation
        model: Model to use (default: claude-haiku-4.5)
        api_key: API key (default: from env or .env file)

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

    # Get API key
    key = api_key or _load_api_key()

    # Prepare API request
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model or DEFAULT_MODEL,
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
