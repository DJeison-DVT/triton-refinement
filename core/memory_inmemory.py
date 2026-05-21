"""In-memory pattern memory implementation (core timeline, weeks 2-3)."""

from core.pattern_memory import PatternMemory


class InMemoryPatternMemory(PatternMemory):
    """List-based accumulation of translation patterns.

    Stores all entries in a plain list. retrieve() returns the most recent
    top_k entries. The context parameter is ignored (no filtering).

    This is the primary implementation for weeks 2-3. A RAG implementation
    (memory_rag.py) will be added in week 4 as a stretch goal.
    """

    def __init__(self) -> None:
        self._entries: list[dict] = []

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
