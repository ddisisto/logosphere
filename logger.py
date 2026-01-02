"""
Structured logging for Logosphere experiments.

Single JSONL file with typed events for streaming and analysis.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any


class Logger:
    def __init__(self, output_dir: Path):
        """
        Initialize logger for experiment.

        Args:
            output_dir: Directory for log files (typically experiments/run-name/logs/)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "experiment.jsonl"

        # Open file in append mode
        self.file_handle = open(self.log_file, 'a')

    def _write_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Write typed event to JSONL log."""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        self.file_handle.write(json.dumps(event) + '\n')
        self.file_handle.flush()  # Ensure streaming writes

    def log_experiment_start(self, config: dict[str, Any], init_signature: str, num_seeds: int) -> None:
        """
        Log experiment initialization.

        Args:
            config: Experiment configuration parameters
            init_signature: Signature from init.md
            num_seeds: Number of seed messages loaded
        """
        self._write_event("experiment_start", {
            "config": config,
            "init_signature": init_signature,
            "num_seed_messages": num_seeds
        })

    def log_round_start(self, round_num: int, pool_size: int, active_pool_size: int) -> None:
        """
        Log start of round.

        Args:
            round_num: Current round number
            pool_size: Total pool size
            active_pool_size: Active pool size (tail M)
        """
        self._write_event("round_start", {
            "round": round_num,
            "pool_size": pool_size,
            "active_pool_size": active_pool_size
        })

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
        """
        Log Mind invocation and output.

        Args:
            round_num: Current round number
            mind_id: Mind identifier (0 to N-1)
            input_messages: Messages sampled for this Mind
            thinking: Private reasoning (before first ---)
            transmitted: Messages successfully transmitted to pool
            completed: Whether output properly terminated
            signature: Context signature (if any)
            tokens_used: Completion tokens consumed
            raw_output: Full raw LLM output
        """
        self._write_event("mind_invocation", {
            "round": round_num,
            "mind_id": mind_id,
            "input_messages": input_messages,
            "thinking": thinking,
            "transmitted": transmitted,
            "completed": completed,
            "signature": signature,
            "tokens_used": tokens_used,
            "raw_output": raw_output
        })

    def log_round_end(self, round_num: int, messages_added: int, pool_delta: list[str]) -> None:
        """
        Log end of round with pool delta.

        Args:
            round_num: Current round number
            messages_added: Number of messages added this round
            pool_delta: New messages added to pool this round
        """
        self._write_event("round_end", {
            "round": round_num,
            "messages_added": messages_added,
            "pool_delta": pool_delta
        })

    def log_experiment_end(self, total_rounds: int, final_pool_size: int, total_tokens: int) -> None:
        """
        Log experiment completion.

        Args:
            total_rounds: Total rounds completed
            final_pool_size: Final total pool size
            total_tokens: Total tokens consumed
        """
        self._write_event("experiment_end", {
            "total_rounds": total_rounds,
            "final_pool_size": final_pool_size,
            "total_tokens": total_tokens
        })

    def close(self) -> None:
        """Close log file."""
        if hasattr(self, 'file_handle') and self.file_handle:
            self.file_handle.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
