"""Tests for PatternMemory ABC and InMemoryPatternMemory implementation."""

import pytest
from core.pattern_memory import PatternMemory
from core.memory_inmemory import InMemoryPatternMemory


class TestPatternMemoryABC:
    def test_cannot_instantiate_directly(self):
        """PatternMemory ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PatternMemory()


class TestInMemoryPatternMemory:
    def test_starts_empty(self):
        """InMemoryPatternMemory starts empty; retrieve returns []."""
        mem = InMemoryPatternMemory()
        result = mem.retrieve("any context")
        assert result == []

    def test_store_and_retrieve_returns_entry_with_correct_keys(self):
        """store + retrieve returns stored entry with op_name, pattern, outcome keys."""
        mem = InMemoryPatternMemory()
        mem.store(op_name="add", pattern="use tl.load with mask", outcome="pass")
        result = mem.retrieve("any context")
        assert len(result) == 1
        entry = result[0]
        assert "op_name" in entry
        assert "pattern" in entry
        assert "outcome" in entry
        assert entry["op_name"] == "add"
        assert entry["pattern"] == "use tl.load with mask"
        assert entry["outcome"] == "pass"

    def test_multiple_stores_accumulate(self):
        """Multiple stores accumulate entries."""
        mem = InMemoryPatternMemory()
        mem.store("op_a", "pattern_a", "pass")
        mem.store("op_b", "pattern_b", "fail")
        mem.store("op_c", "pattern_c", "pass")
        result = mem.retrieve("any context", top_k=10)
        assert len(result) == 3

    def test_retrieve_respects_top_k(self):
        """retrieve respects top_k parameter (returns at most top_k entries)."""
        mem = InMemoryPatternMemory()
        for i in range(10):
            mem.store(f"op_{i}", f"pattern_{i}", "pass")
        result = mem.retrieve("any context", top_k=3)
        assert len(result) == 3

    def test_retrieve_returns_most_recent_entries(self):
        """retrieve returns most recent entries (last top_k)."""
        mem = InMemoryPatternMemory()
        for i in range(5):
            mem.store(f"op_{i}", f"pattern_{i}", "pass")
        # With top_k=2, should return the last 2 stored entries
        result = mem.retrieve("any context", top_k=2)
        assert len(result) == 2
        assert result[0]["op_name"] == "op_3"
        assert result[1]["op_name"] == "op_4"

    def test_retrieve_top_k_default_is_5(self):
        """retrieve default top_k=5 returns at most 5 entries."""
        mem = InMemoryPatternMemory()
        for i in range(8):
            mem.store(f"op_{i}", f"pattern_{i}", "pass")
        result = mem.retrieve("any context")
        assert len(result) == 5

    def test_retrieve_context_parameter_is_ignored(self):
        """context parameter is ignored (no filtering for in-memory)."""
        mem = InMemoryPatternMemory()
        mem.store("add", "some pattern", "pass")
        result_a = mem.retrieve("context that matches nothing")
        result_b = mem.retrieve("add pattern context")
        assert result_a == result_b

    def test_retrieve_top_k_larger_than_stored(self):
        """retrieve with top_k larger than entries returns all entries."""
        mem = InMemoryPatternMemory()
        mem.store("op_a", "pattern_a", "pass")
        mem.store("op_b", "pattern_b", "fail")
        result = mem.retrieve("context", top_k=100)
        assert len(result) == 2

    def test_is_instance_of_pattern_memory(self):
        """InMemoryPatternMemory is a subclass of PatternMemory."""
        mem = InMemoryPatternMemory()
        assert isinstance(mem, PatternMemory)
