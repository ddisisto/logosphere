"""
Configuration parameters for Logosphere experiment.
"""

import os
from pathlib import Path
from datetime import datetime


# Load environment variables
def load_api_key() -> str:
    """Load API key from .env file."""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith('OPENROUTER_API'):
                    return line.split(':', 1)[1].strip()
    raise ValueError("API key not found in .env")


# Directory structure
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


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


# Real-time detection defaults (opt-in for backward compatibility)
DEFAULT_EMBEDDING_CONFIG = {
    'enabled': False,
    'model': 'openai/text-embedding-3-small',
    'fail_mode': 'abort',  # 'abort' or 'skip'
}

DEFAULT_ATTRACTOR_CONFIG = {
    'enabled': False,
    'algorithm': 'hdbscan',
    'min_cluster_size': 5,
    'cluster_active_pool_only': True,
}

DEFAULT_INTERVENTION_CONFIG = {
    'enabled': False,
    'strategy': 'none',
}


# System prompt - minimal functional framing
SYSTEM_PROMPT = """You receive messages from others. Read them.

Content before the first --- is private thinking, not transmitted.
After ---, write messages for others, separated by ---."""


def load_experiment_config(exp_dir: Path) -> dict:
    """
    Load experiment config, merging with defaults.

    Args:
        exp_dir: Experiment directory containing config.json

    Returns:
        Complete config dict with all sections
    """
    import json

    config_path = exp_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {exp_dir}")

    with open(config_path) as f:
        config = json.load(f)

    # Merge with defaults for new sections
    if 'embeddings' not in config:
        config['embeddings'] = DEFAULT_EMBEDDING_CONFIG.copy()
    else:
        merged = DEFAULT_EMBEDDING_CONFIG.copy()
        merged.update(config['embeddings'])
        config['embeddings'] = merged

    if 'attractor_detection' not in config:
        config['attractor_detection'] = DEFAULT_ATTRACTOR_CONFIG.copy()
    else:
        merged = DEFAULT_ATTRACTOR_CONFIG.copy()
        merged.update(config['attractor_detection'])
        config['attractor_detection'] = merged

    if 'interventions' not in config:
        config['interventions'] = DEFAULT_INTERVENTION_CONFIG.copy()
    else:
        merged = DEFAULT_INTERVENTION_CONFIG.copy()
        merged.update(config['interventions'])
        config['interventions'] = merged

    return config


def create_experiment_dir(name: str = None) -> Path:
    """
    Create experiment directory with timestamp.

    Args:
        name: Optional experiment name (default: timestamp)

    Returns:
        Path to experiment directory
    """
    if name is None:
        name = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    exp_dir = EXPERIMENTS_DIR / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    return exp_dir
