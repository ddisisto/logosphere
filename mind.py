"""
Mind invocation and message parsing for Logosphere experiment.

Each Mind is a single stateless API call.
"""

import requests
from config import API_KEY, MODEL, API_BASE_URL, DELIMITER, TOKEN_LIMIT


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


def parse_output(raw: str, signature_max_len: int = 32) -> tuple[str, list[str], bool, str]:
    """
    Parse Mind output into components.

    Rules:
    - Split on DELIMITER (---) on its own line
    - First block = thinking (not transmitted)
    - Each subsequent non-blank block = message
    - First blank block = termination (completed=True)
    - Final unterminated DELIMITER = signature (appended to all messages)
    - No blank block before end = incomplete (completed=False, nothing transmitted)

    Returns:
        (thinking, messages, completed, signature)

    where:
        thinking: str - private reasoning before first ---
        messages: list[str] - messages to transmit
        completed: bool - True if properly terminated with blank block
        signature: str - context signature if present (truncated to signature_max_len)
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
    thinking = blocks[0].lstrip('\n') if blocks else ""  # Remove any leading newline we added

    # Process blocks as messages, looking for termination
    messages = []
    completed = False
    termination_index = -1

    for i, block in enumerate(blocks[1:], start=1):  # Skip thinking block
        stripped = block.strip()

        # Blank block = termination
        if not stripped:
            completed = True
            termination_index = i
            break

        # Non-blank block = message
        messages.append(block)

    # Check for signature ONLY if output completed
    # Signature = non-empty content after blank terminator
    signature = ""
    if completed and termination_index < len(blocks) - 1:
        # There's content after the termination block
        potential_sig = blocks[termination_index + 1].strip()
        if potential_sig and potential_sig != DELIMITER:
            signature = potential_sig[:signature_max_len]

    # If not completed, drop all messages and signature
    if not completed:
        messages = []
        signature = ""

    # Append signature to all transmitted messages
    if signature and messages:
        messages = [f"{msg}\n\n{signature}" for msg in messages]

    return thinking, messages, completed, signature


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
            'completed': bool,
            'signature': str,
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
    thinking, transmitted, completed, signature = parse_output(raw_output)

    return {
        'thinking': thinking,
        'transmitted': transmitted,
        'completed': completed,
        'signature': signature,
        'tokens_used': tokens_used,
        'raw_output': raw_output
    }
