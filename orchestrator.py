"""
Orchestrator - coordinates rounds, Minds, and pool for Logosphere experiments.
"""

from pool import Pool
from logger import Logger
from mind import invoke_mind
import config


class Orchestrator:
    def __init__(
        self,
        pool: Pool,
        logger: Logger,
        n_minds: int = None,
        k_samples: int = None,
        system_prompt: str = None,
        token_limit: int = None
    ):
        """
        Initialize orchestrator.

        Args:
            pool: Message pool
            logger: Experiment logger
            n_minds: Number of minds (defaults to config.N_MINDS)
            k_samples: Samples per mind (defaults to config.K_SAMPLES)
            system_prompt: System prompt (defaults to config.SYSTEM_PROMPT)
            token_limit: Token limit per mind (defaults to config.TOKEN_LIMIT)
        """
        self.pool = pool
        self.logger = logger
        self.n_minds = n_minds if n_minds is not None else config.N_MINDS
        self.k_samples = k_samples if k_samples is not None else config.K_SAMPLES
        self.system_prompt = system_prompt if system_prompt is not None else config.SYSTEM_PROMPT
        self.token_limit = token_limit if token_limit is not None else config.TOKEN_LIMIT
        self.total_tokens = 0

    def run_round(self, round_num: int) -> int:
        """
        Execute one round - invoke all Minds sequentially.

        Args:
            round_num: Current round number (1-indexed)

        Returns:
            Number of messages added to pool this round
        """
        # Track messages added this round for pool delta
        round_messages = []

        # Log round start
        self.logger.log_round_start(
            round_num=round_num,
            pool_size=self.pool.size(),
            active_pool_size=self.pool.active_size()
        )

        # Invoke each Mind sequentially
        for mind_id in range(self.n_minds):
            # Sample K messages from active pool
            sampled_messages = self.pool.sample(self.k_samples)

            # Invoke Mind (fail fast on errors)
            result = invoke_mind(
                system_prompt=self.system_prompt,
                messages=sampled_messages,
                token_limit=self.token_limit
            )

            # Track tokens
            self.total_tokens += result['tokens_used']

            # Log Mind invocation
            self.logger.log_mind_invocation(
                round_num=round_num,
                mind_id=mind_id,
                input_messages=sampled_messages,
                thinking=result['thinking'],
                transmitted=result['transmitted'],
                tokens_used=result['tokens_used'],
                raw_output=result['raw_output']
            )

            # Add transmitted messages to pool
            for msg in result['transmitted']:
                self.pool.add_message(msg)
                round_messages.append(msg)

        # Log round end with pool delta
        self.logger.log_round_end(
            round_num=round_num,
            messages_added=len(round_messages),
            pool_delta=round_messages
        )

        return len(round_messages)

    def run(self, max_rounds: int) -> None:
        """
        Execute full experiment.

        Args:
            max_rounds: Number of rounds to run
        """
        for round_num in range(1, max_rounds + 1):
            messages_added = self.run_round(round_num)

            # Could add stopping conditions here if needed
            # For now, just run all rounds
