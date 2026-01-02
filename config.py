"""
Configuration parameters for Logosphere experiment.
"""

import os
from pathlib import Path


# Load environment variables
def load_api_key() -> str:
    """Load API key from .env file."""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith('OPENROUTER_API'):
                    return line.split(':', 1)[1].strip()
    raise ValueError("API key not found in .env")


# Population parameters
N_MINDS = 3  # Start small - 3 minds
K_SAMPLES = 3  # Each mind sees 3 messages
M_ACTIVE_POOL = 15  # Active pool of 15 messages (5x samples)
MAX_ROUNDS = 10  # Initial run - 10 rounds
TOKEN_LIMIT = 2000  # Per mind per round


# API configuration
API_KEY = load_api_key()
MODEL = "anthropic/claude-3.5-sonnet"  # Via OpenRouter
API_BASE_URL = "https://openrouter.ai/api/v1"


# Message delimiter
DELIMITER = "---"


# System prompt - minimal functional framing
SYSTEM_PROMPT = """You receive messages from others. Read them.
You may write messages to share with others.

Format: Messages separated by --- on its own line.
To finish, write a blank message (two --- with nothing between)."""


# Seed messages - bootstrap the pool
SEED_MESSAGES = [
    "What patterns persist?",
    "Attention is the resource.",
    "Simple rules, complex outcomes."
]
