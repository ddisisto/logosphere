# CLAUDE.md - Implementation Specification

## Experimental Boundary

**INSIDE THE EXPERIMENT:**
- Messages in the pool
- Mind inputs (sampled messages + system prompt)
- Mind outputs (parsed messages)

**OUTSIDE THE EXPERIMENT:**
- Orchestrator (scheduler, sampler, parser)
- API calls to LLM
- Logging system
- File I/O

**Critical:** Nothing inside the experiment can see round numbers, timestamps, authorship, or any metadata. Messages are anonymous and timeless from the perspective of Minds.

---

## Implementation Structure

```
logosphere/
├── orchestrator.py    # Main loop, coordination
├── pool.py           # Message storage and sampling
├── mind.py           # API invocation and parsing
├── logger.py         # Structured logging
├── config.py         # Parameters and system prompt
└── run.py            # Entry point
```

---

## Experiment Directory Structure

Each experiment run is self-contained:

```
logosphere/
├── pool.py, mind.py, etc.        # Core implementation
├── init-template.txt             # Template for seeding
└── experiments/
    ├── 2026-01-02-first-run/
    │   ├── init.md               # Seed messages (parsed like Mind output)
    │   ├── config.json           # Parameter snapshot
    │   └── logs/                 # JSONL experiment logs
    └── 2026-01-02-second-run/
        └── ...
```

### init.md Format

Seeds the pool by reusing Mind parsing:
- Pre-delimiter content: notes, metadata (not transmitted)
- Messages: standard `---` separated blocks
- Termination: blank message required
- Signature: `~init` (appended to all seed messages)

**Validation:** Mind outputs with signature exactly matching `~init` are invalid and discarded (prevents impersonation).

**Workflow:**
1. Copy `init-template.txt` to `experiments/run-name/init.md`
2. Customize seed messages
3. Run experiment (init.md parsed to seed pool)

---

## Sub-Task 1: Core Data Structures

**Deliverable:** `pool.py`, `config.py`, `init_parser.py`

### pool.py

```python
class Pool:
    def __init__(self, max_active: int):
        """
        max_active: M (tail size for active pool)
        """
        pass

    def add_message(self, content: str) -> None:
        """Add message to pool (appends to history)"""
        pass

    def sample(self, k: int) -> list[str]:
        """Sample k messages uniformly from active pool (tail M)"""
        pass

    def get_active(self) -> list[str]:
        """Return active pool (tail M messages)"""
        pass

    def get_all(self) -> list[str]:
        """Return full history"""
        pass

    def size(self) -> int:
        """Return total messages in history"""
        pass
```

### config.py

```python
# Directory structure
PROJECT_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
INIT_TEMPLATE = PROJECT_ROOT / "init-template.txt"

# Population parameters
N_MINDS = 3
K_SAMPLES = 3
M_ACTIVE_POOL = 15
MAX_ROUNDS = 10
TOKEN_LIMIT = 2000

# API configuration
API_KEY = load_api_key()  # from .env
MODEL = "anthropic/claude-3.5-sonnet"
API_BASE_URL = "https://openrouter.ai/api/v1"

# System prompt
SYSTEM_PROMPT = """You receive messages from others. Read them.
You may write messages to share with others.

Format: Messages separated by --- on its own line.
To finish, write a blank message (two --- with nothing between)."""

# Init signature for validation
INIT_SIGNATURE = "~init"

# Utilities
def create_experiment_dir(name: str = None) -> Path:
    """Create timestamped experiment directory"""
```

### init_parser.py

```python
def load_init_file(init_path: Path) -> tuple[list[str], str]:
    """Parse init.md to extract seed messages and signature"""
```

**Note:** No signature validation/filtering. All Mind outputs go to pool regardless of signature. Let the memes meme.

**Verification:** Can create pool, add messages, sample correctly. Active pool is tail M. Can parse init.md.

---

## Sub-Task 2: Mind Invocation

**Deliverable:** `mind.py`

```python
def invoke_mind(system_prompt: str, messages: list[str], token_limit: int) -> dict:
    """
    Invokes LLM with formatted input.

    Returns:
    {
        'thinking': str,           # Private reasoning (before first ---)
        'transmitted': list[str],  # Messages to pool
        'completed': bool,         # True if terminated with blank block
        'tokens_used': int,
        'raw_output': str
    }
    """
    pass

def format_input(system_prompt: str, messages: list[str]) -> str:
    """
    Format: system_prompt + delimiter + messages separated by delimiter
    """
    pass

def parse_output(raw: str) -> tuple[str, list[str], bool]:
    """
    Parse output into (thinking, messages, completed).

    Rules:
    - Everything before first --- is thinking
    - Each non-blank block between --- is a message
    - First blank block terminates (completed=True)
    - No termination = completed=False, nothing transmitted
    """
    pass
```

**Verification:** Can invoke API, parse output correctly, handle termination.

---

## Sub-Task 3: Logging

**Deliverable:** `logger.py`

```python
class Logger:
    def __init__(self, output_dir: Path):
        """Initialize logger - creates experiment.jsonl in output_dir"""

    def log_experiment_start(self, config: dict, init_signature: str, num_seeds: int) -> None:
        """Log experiment initialization"""

    def log_round_start(self, round_num: int, pool_size: int, active_pool_size: int) -> None:
        """Log round start with pool state"""

    def log_mind_invocation(
        self,
        round_num: int,
        mind_id: int,
        input_messages: list[str],
        thinking: str,
        transmitted: list[str],
        completed: bool,
        signature: str,
        tokens_used: int,
        raw_output: str
    ) -> None:
        """Log Mind invocation and output"""

    def log_round_end(self, round_num: int, messages_added: int, pool_delta: list[str]) -> None:
        """Log round end with pool delta (new messages added)"""

    def log_experiment_end(self, total_rounds: int, final_pool_size: int, total_tokens: int) -> None:
        """Log experiment completion"""
```

**Format:** Single JSONL file (`experiment.jsonl`) with typed events:
- `experiment_start` - config, init signature, seed count
- `round_start` - round number, pool sizes
- `mind_invocation` - full Mind I/O and metadata
- `round_end` - pool delta (messages added this round)
- `experiment_end` - summary statistics

**Design:** Streaming writes with flush, pool deltas (not snapshots), ISO timestamps, context manager support.

**Verification:** Logs written correctly, typed events parseable, complete data for reconstruction.

---

## Sub-Task 4: Orchestrator

**Deliverable:** `orchestrator.py`

```python
class Orchestrator:
    def __init__(self, config, pool, logger):
        pass

    def run_round(self, round_num: int) -> int:
        """
        Execute one round:
        1. For each mind (0 to N-1):
           - Sample K messages from pool
           - Invoke mind
           - Parse output
           - Add transmitted messages to pool
           - Log interaction

        Returns: number of messages added to pool
        """
        pass

    def run(self, max_rounds: int) -> None:
        """
        Execute experiment:
        1. Initialize pool with seed messages
        2. For each round until max_rounds:
           - Run round
           - Save pool snapshot
           - Check stopping conditions
        """
        pass
```

**Verification:** Can run multiple rounds, pool grows, logs are complete.

---

## Sub-Task 5: Entry Point

**Deliverable:** `run.py`

```python
#!/usr/bin/env python3
"""
Entry point for Logosphere experiment.

Usage:
    python run.py [--rounds N] [--output-dir PATH]
"""

if __name__ == "__main__":
    # Parse args
    # Load config
    # Initialize pool with seeds
    # Initialize logger
    # Create orchestrator
    # Run experiment
    # Print summary
```

**Verification:** Can run end-to-end experiment, produces logs and results.

---

## Parameter Selection (Initial Values)

**To be determined in Sub-Task 1:**

| Parameter | Initial Value | Rationale |
|-----------|---------------|-----------|
| N_MINDS | 3-5 | Small enough to observe, large enough for dynamics |
| K_SAMPLES | 3 | Attention budget - not too narrow, not overwhelming |
| M_ACTIVE_POOL | 15-20 | ~3-4x the samples, allows some diversity |
| MAX_ROUNDS | 10-20 | Enough to see trends, not too expensive |
| TOKEN_LIMIT | 2000 | Reasonable thinking + message space |

**System prompt:** Absolute minimum viable. Something like:

```
You receive messages from others. Read them.
You may write messages to share with others.

Format: Messages separated by --- on its own line.
To finish, write a blank message (two --- with nothing between).
```

**Seed message(s):** TBD - single clear proposition or question.

---

## Verification Protocol

**After each sub-task:**
1. Code is written
2. Basic tests verify behavior
3. I review before proceeding to next sub-task

**After Sub-Task 5:**
1. Run full experiment with initial parameters
2. Verify logs are complete and readable
3. Inspect first few rounds manually
4. Confirm experimental boundary is maintained (no metadata leakage)

---

## What This Specification Omits

**Deferred to post-MVP:**
- Analysis tools
- Visualization
- Parameter optimization
- Advanced sampling strategies
- Heterogeneous minds
- Multi-turn minds

**Focus:** Get the minimal system running and observable first.

---

## Success Criteria

**Sub-task level:** Each component works in isolation.

**Integration level:** Full experiment runs without errors.

**Scientific level:** We can read logs and see what happened in the pool over time.

If we achieve scientific level success, we have a working substrate for memetic dynamics experiments.
