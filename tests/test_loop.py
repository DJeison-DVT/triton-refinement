"""Tests for core/loop.py — refinement loop with trajectory logging."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.loop import (
    IterationLog,
    RefinementResult,
    generate_with_refinement,
    save_trajectory,
)
from core.memory_inmemory import InMemoryPatternMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PYTORCH_CODE = "def add(x, y):\n    return x + y\n"
# When grammar is provided the kernel output is used raw; wrapper goes through extract_code.
# For free-form (no grammar) the single response goes through extract_code, which adds "\n".


def _mock_client(responses: list[str]) -> MagicMock:
    """Return a MagicMock LLMClient whose generate() iterates through responses."""
    client = MagicMock()
    client.generate = MagicMock(side_effect=responses)
    return client


# Minimal Triton-ish code used in responses.
_KERNEL = "@triton.jit\ndef kernel(x_ptr, BLOCK: tl.constexpr):\n    pass\n"
_WRAPPER = "def add_triton(x, y):\n    kernel[(1,)](x)\n    return x\n"
_FULL = _KERNEL + "\n" + _WRAPPER

# When no grammar: extract_code strips any fences; plain text → adds trailing "\n".
# Use plain code strings so extract_code returns them unchanged (already end with "\n").


# ---------------------------------------------------------------------------
# Test 1: Happy path — translate → compile ok → test ok → review approves
# ---------------------------------------------------------------------------


@patch("core.loop._run_code", return_value=(True, ""))
def test_happy_path(mock_run):
    """No grammar: one LLM call → compile → test → review APPROVED → passed=True."""
    client = _mock_client([_FULL, "APPROVED"])
    result = generate_with_refinement(
        "add", PYTORCH_CODE, "assert True\n", client, max_iters=5
    )
    assert result.passed is True
    assert result.total_iterations == 1
    # generate called twice: translate + review
    assert client.generate.call_count == 2


# ---------------------------------------------------------------------------
# Test 2: Grammar two-step — kernel + wrapper calls
# ---------------------------------------------------------------------------


@patch("core.loop._run_code", return_value=(True, ""))
def test_grammar_two_step(mock_run):
    """With grammar: kernel call (grammar) + wrapper call → combined code."""
    # kernel response is raw (no extraction), wrapper goes through extract_code
    client = _mock_client([_KERNEL, _WRAPPER, "APPROVED"])
    result = generate_with_refinement(
        "add", PYTORCH_CODE, "assert True\n", client, grammar="fake", max_iters=5
    )
    assert result.passed is True
    # final_code must contain content from both kernel and wrapper
    assert "@triton.jit" in result.final_code
    assert "add_triton" in result.final_code
    # First generate call passed grammar kwarg
    first_call_kwargs = client.generate.call_args_list[0][1]
    assert first_call_kwargs.get("grammar") == "fake"
    # Second generate call (wrapper) has no grammar
    second_call_kwargs = client.generate.call_args_list[1][1]
    assert second_call_kwargs.get("grammar") is None


# ---------------------------------------------------------------------------
# Test 3: Compile failure → fix → success
# ---------------------------------------------------------------------------


def test_compile_failure_then_fix():
    """First _run_code fails (compile), fix is called, second run passes → passed=True."""
    run_side_effects = [
        (False, "SyntaxError: invalid syntax"),  # compile iter 1
        (True, ""),                               # compile iter 2
        (True, ""),                               # test iter 2
    ]
    # LLM responses: translate, fix, review
    client = _mock_client([_FULL, _FULL, "APPROVED"])

    with patch("core.loop._run_code", side_effect=run_side_effects):
        result = generate_with_refinement(
            "add", PYTORCH_CODE, "assert True\n", client, max_iters=5
        )

    assert result.passed is True
    assert result.total_iterations == 2


# ---------------------------------------------------------------------------
# Test 4: Test failure → fix → success
# ---------------------------------------------------------------------------


def test_test_failure_then_fix():
    """Compile passes, test fails, fix called, second iteration passes → passed=True."""
    run_side_effects = [
        (True, ""),                                        # compile iter 1
        (False, "AssertionError: values differ"),         # test iter 1
        (True, ""),                                        # compile iter 2
        (True, ""),                                        # test iter 2
    ]
    client = _mock_client([_FULL, _FULL, "APPROVED"])

    with patch("core.loop._run_code", side_effect=run_side_effects):
        result = generate_with_refinement(
            "add", PYTORCH_CODE, "assert True\n", client, max_iters=5
        )

    assert result.passed is True
    assert result.total_iterations == 2


# ---------------------------------------------------------------------------
# Test 5: Review rejection → fix → success
# ---------------------------------------------------------------------------


def test_review_rejection_then_fix():
    """Compile ok, test ok, reviewer rejects, fix, retry approved → passed=True."""
    run_side_effects = [
        (True, ""),  # compile iter 1
        (True, ""),  # test iter 1
        (True, ""),  # compile iter 2
        (True, ""),  # test iter 2
    ]
    # translate, review-reject, fix, review-approve
    client = _mock_client([_FULL, "Missing masks in tl.load", _FULL, "APPROVED"])

    with patch("core.loop._run_code", side_effect=run_side_effects):
        result = generate_with_refinement(
            "add", PYTORCH_CODE, "assert True\n", client, max_iters=5
        )

    assert result.passed is True
    assert result.total_iterations == 2


# ---------------------------------------------------------------------------
# Test 6: Max iterations exhausted
# ---------------------------------------------------------------------------


@patch("core.loop._run_code", return_value=(False, "SyntaxError"))
def test_max_iterations_exhausted(mock_run):
    """_run_code always fails → loop exhausts max_iters → passed=False."""
    max_iters = 3
    # translate + one fix per iteration
    client = _mock_client([_FULL] + [_FULL] * max_iters)
    result = generate_with_refinement(
        "add", PYTORCH_CODE, "assert True\n", client, max_iters=max_iters
    )
    assert result.passed is False
    assert result.total_iterations == max_iters


# ---------------------------------------------------------------------------
# Test 7: Trajectory contains expected stages
# ---------------------------------------------------------------------------


@patch("core.loop._run_code", return_value=(True, ""))
def test_trajectory_stages(mock_run):
    """Trajectory must contain 'translate', 'compile', 'test', 'review' stages."""
    client = _mock_client([_FULL, "APPROVED"])
    result = generate_with_refinement(
        "add", PYTORCH_CODE, "assert True\n", client, max_iters=5
    )
    stages = {entry.stage for entry in result.trajectory}
    assert "translate" in stages
    assert "compile" in stages
    assert "test" in stages
    assert "review" in stages


# ---------------------------------------------------------------------------
# Test 8: Trajectory entries have non-empty timestamps
# ---------------------------------------------------------------------------


@patch("core.loop._run_code", return_value=(True, ""))
def test_trajectory_timestamps(mock_run):
    """Each trajectory entry must have a non-empty timestamp string."""
    client = _mock_client([_FULL, "APPROVED"])
    result = generate_with_refinement(
        "add", PYTORCH_CODE, "assert True\n", client, max_iters=5
    )
    for entry in result.trajectory:
        assert isinstance(entry.timestamp, str)
        assert len(entry.timestamp) > 0


# ---------------------------------------------------------------------------
# Test 9: Pattern memory stored on success
# ---------------------------------------------------------------------------


@patch("core.loop._run_code", return_value=(True, ""))
def test_pattern_memory_on_success(mock_run):
    """On success, pattern_memory.store is called with outcome='pass'."""
    client = _mock_client([_FULL, "APPROVED"])
    mem = InMemoryPatternMemory()
    generate_with_refinement(
        "add", PYTORCH_CODE, "assert True\n", client, max_iters=5, pattern_memory=mem
    )
    entries = mem.retrieve("any", top_k=10)
    assert len(entries) == 1
    assert entries[0]["outcome"] == "pass"
    assert entries[0]["op_name"] == "add"


# ---------------------------------------------------------------------------
# Test 10: Pattern memory stored on failure
# ---------------------------------------------------------------------------


@patch("core.loop._run_code", return_value=(False, "SyntaxError"))
def test_pattern_memory_on_failure(mock_run):
    """On failure (exhausted), pattern_memory.store is called with outcome='fail'."""
    max_iters = 2
    client = _mock_client([_FULL] + [_FULL] * max_iters)
    mem = InMemoryPatternMemory()
    result = generate_with_refinement(
        "add", PYTORCH_CODE, "assert True\n", client, max_iters=max_iters, pattern_memory=mem
    )
    assert result.passed is False
    entries = mem.retrieve("any", top_k=10)
    assert len(entries) == 1
    assert entries[0]["outcome"] == "fail"
    assert entries[0]["op_name"] == "add"


# ---------------------------------------------------------------------------
# Test 11: save_trajectory writes valid JSONL
# ---------------------------------------------------------------------------


def test_save_trajectory_writes_valid_jsonl():
    """save_trajectory writes a .jsonl file with one JSON object per line."""
    trajectory = [
        IterationLog(
            iteration=0,
            stage="translate",
            prompt_hash="abc123",
            generation="@triton.jit\ndef k(): pass\n",
            outcome="pass",
            error=None,
            timestamp="2026-05-20T00:00:00+00:00",
        ),
        IterationLog(
            iteration=1,
            stage="compile",
            prompt_hash="",
            generation="@triton.jit\ndef k(): pass\n",
            outcome="fail",
            error="SyntaxError",
            timestamp="2026-05-20T00:00:01+00:00",
        ),
    ]
    result = RefinementResult(
        op_name="test_op",
        final_code="@triton.jit\ndef k(): pass\n",
        test_code="assert True\n",
        passed=False,
        total_iterations=1,
        trajectory=trajectory,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = save_trajectory(result, Path(tmp_dir))
        assert out_path.exists()
        assert out_path.name == "test_op.jsonl"

        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

        expected_keys = {
            "op_name", "iteration", "stage", "prompt_hash",
            "generation", "outcome", "error", "timestamp",
        }
        for line in lines:
            obj = json.loads(line)
            assert expected_keys == set(obj.keys())
            assert obj["op_name"] == "test_op"
