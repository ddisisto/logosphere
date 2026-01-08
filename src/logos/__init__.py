"""
Logos - Pool-based reasoning with session infrastructure.

New entrypoint with fork/rollback/comparison built in from the start.
"""

from .config import LogosConfig
from .runner import LogosRunner

__all__ = ["LogosConfig", "LogosRunner"]
