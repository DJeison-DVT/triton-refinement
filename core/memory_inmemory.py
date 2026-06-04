"""In-memory pattern memory implementation."""

from core.pattern_memory import PatternMemory
from core.triton_patterns import TRITON_PATTERNS


class InMemoryPatternMemory(PatternMemory):
    """List-based accumulation of translation patterns.

    Stores all entries in a plain list. retrieve() returns the most recent
    top_k entries. The context parameter is ignored (no filtering).

    Can be pre-seeded with curated Triton best practices via seed=True.
    """

    def __init__(self) -> None:
        self._entries: list[dict] = list(TRITON_PATTERNS)

    def store(self, op_name: str, pattern: str, outcome: str) -> None:
        """Append a pattern entry to the in-memory list.

        Args:
            op_name: Name of the PyTorch operator being translated.
            pattern: Description of the pattern used or observed.
            outcome: Result of applying the pattern (e.g., "pass", "fail").
        """
        self._entries.append(
            {"op_name": op_name, "pattern": pattern, "outcome": outcome}
        )

    def retrieve(self, context: str, top_k: int = 5) -> list[dict]:
        """Return the most recent top_k entries.

        The context parameter is ignored; no semantic filtering is performed.

        Args:
            context: Ignored for this implementation.
            top_k: Maximum number of entries to return.

        Returns:
            List of the most recently stored dicts (up to top_k),
            ordered from oldest to newest among the selected slice.
        """
        return self._entries[-top_k:] if self._entries else []
