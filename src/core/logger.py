"""
Structured logging for Logosphere experiments.

Single JSONL file with typed events for streaming and analysis.

Event types:
- experiment_start: Config, seeds
- round_start: Pool state
- mind_invocation: Full I/O (legacy: includes text, new: references vector IDs)
- round_end: Messages added
- attractor_state: Clustering results (new)
- embedding_batch: Embedding performance (new)
- error: Abort scenarios (new)
- experiment_end: Summary stats
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional


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
            transmitted: Messages transmitted to pool
            tokens_used: Completion tokens consumed
            raw_output: Full raw LLM output
        """
        self._write_event("mind_invocation", {
            "round": round_num,
            "mind_id": mind_id,
            "input_messages": input_messages,
            "thinking": thinking,
            "transmitted": transmitted,
            "tokens_used": tokens_used,
            "raw_output": raw_output
        })

    def log_round_end(
        self,
        round_num: int,
        messages_added: int,
        pool_delta: Optional[list[str]] = None,
        vector_ids: Optional[list[int]] = None,
    ) -> None:
        """
        Log end of round with pool delta.

        Args:
            round_num: Current round number
            messages_added: Number of messages added this round
            pool_delta: New messages added (legacy, for backward compat)
            vector_ids: Vector IDs of added messages (new VectorDB mode)
        """
        data = {
            "round": round_num,
            "messages_added": messages_added,
        }
        if pool_delta is not None:
            data["pool_delta"] = pool_delta
        if vector_ids is not None:
            data["vector_ids"] = vector_ids

        self._write_event("round_end", data)

    def log_attractor_state(
        self,
        round_num: int,
        num_clusters: int,
        noise_count: int,
        clusters: list[dict],
        detected: bool = True,
    ) -> None:
        """
        Log attractor detection results.

        Args:
            round_num: Current round number
            num_clusters: Number of clusters found
            noise_count: Points not in any cluster
            clusters: Cluster details (id, size, coherence, representative)
            detected: Whether attractors were detected
        """
        # Simplify cluster data for logging (exclude centroid vectors)
        logged_clusters = []
        for c in clusters:
            logged_clusters.append({
                "id": c["id"],
                "size": c["size"],
                "coherence": c.get("coherence"),
                "representative": c.get("representative", "")[:200],  # Truncate
                "rounds": c.get("rounds", []),
            })

        self._write_event("attractor_state", {
            "round": round_num,
            "detected": detected,
            "num_clusters": num_clusters,
            "noise_count": noise_count,
            "clusters": logged_clusters,
        })

    def log_embedding_batch(
        self,
        round_num: int,
        num_texts: int,
        latency_ms: float,
        model: str,
    ) -> None:
        """
        Log embedding batch performance.

        Args:
            round_num: Current round number
            num_texts: Number of texts embedded
            latency_ms: API call latency in milliseconds
            model: Embedding model used
        """
        self._write_event("embedding_batch", {
            "round": round_num,
            "num_texts": num_texts,
            "latency_ms": latency_ms,
            "model": model,
        })

    def log_error(
        self,
        message: str,
        round_num: Optional[int] = None,
        error_type: str = "error",
    ) -> None:
        """
        Log error event (may trigger experiment abort).

        Args:
            message: Error description
            round_num: Round where error occurred (if applicable)
            error_type: Error category (error, warning, abort)
        """
        data = {
            "message": message,
            "error_type": error_type,
        }
        if round_num is not None:
            data["round"] = round_num

        self._write_event("error", data)

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
