"""Abstract base class for pattern memory backends."""

from abc import ABC, abstractmethod


class PatternMemory(ABC):
    """Interface for storing and retrieving translation patterns.

    Allows the reviewer prompt to include patterns from previous translations.
    Implementations are swappable via --pattern-memory flag.
    """

    @abstractmethod
    def store(self, op_name: str, pattern: str, outcome: str) -> None:
        """Store a pattern observed during a translation attempt.

        Args:
            op_name: Name of the PyTorch operator being translated.
            pattern: Description of the pattern used or observed.
            outcome: Result of applying the pattern (e.g., "pass", "fail").
        """
        ...

    @abstractmethod
    def retrieve(self, context: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant patterns for a given context.

        Args:
            context: Context string (e.g., PyTorch source code) used to
                     select relevant patterns. May be ignored by some backends.
            top_k: Maximum number of patterns to return.

        Returns:
            List of dicts with keys: op_name, pattern, outcome.
            Most recent entries preferred.
        """
        ...
