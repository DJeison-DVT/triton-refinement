"""Tests for core/loop.py — refinement loop with trajectory logging."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.loop import (
    IterationLog,
    RefinementResult,
    _extract_code,
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


def _mock_client(
    complete_responses: list[str],
    generate_responses: list[str] | None = None,
) -> MagicMock:
    """Return a MagicMock LLMClient with separate complete/generate responses.

    complete_responses: responses for the completion API (translation).
    generate_responses: responses for the chat API (fix + review).
    """
    client = MagicMock()
    client.complete = MagicMock(side_effect=complete_responses)
    client.generate = MagicMock(
        side_effect=generate_responses if generate_responses else []
    )
    return client


# Minimal Triton-ish code used in responses.
_KERNEL = "@triton.jit\ndef kernel(x_ptr, BLOCK: tl.constexpr):\n    pass\n"
_WRAPPER = "def add_triton(x, y):\n    kernel[(1,)](x)\n    return x\n"
_FULL = _KERNEL + "\n" + _WRAPPER
# Distinct "fixed" version so dedup check doesn't trigger in tests
_FIXED_CODE = "@triton.jit\ndef kernel(x_ptr, BLOCK: tl.constexpr):\n    pass\n\ndef add_triton(x, y):\n    kernel[(1,)](x)\n    return x  # fixed\n"
_FIXED = f"```python\n{_FIXED_CODE}```"

# When no grammar: extract_code strips any fences; plain text → adds trailing "\n".
# Use plain code strings so extract_code returns them unchanged (already end with "\n").


# ---------------------------------------------------------------------------
# Test: _extract_code helper
# ---------------------------------------------------------------------------


def test_extract_code_from_python_fence():
    """Extract code from a ```python fence."""
    raw = "Here is the fix:\n```python\nimport torch\ndef foo():\n    pass\n```\nDone."
    assert _extract_code(raw) == "import torch\ndef foo():\n    pass"


def test_extract_code_no_fence():
    """When no fence, return the raw response stripped."""
    raw = "import torch\ndef foo():\n    pass\n"
    assert _extract_code(raw) == "import torch\ndef foo():\n    pass"


def test_extract_code_bare_fence():
    """Extract code from a bare ``` fence (no language tag)."""
    raw = "```\nimport torch\ndef foo():\n    pass\n```"
    assert _extract_code(raw) == "import torch\ndef foo():\n    pass"


# ---------------------------------------------------------------------------
# Test 1: Happy path — translate → compile ok → test ok → review approves
# ---------------------------------------------------------------------------


@patch("core.loop._run_code", return_value=(True, ""))
def test_happy_path(mock_run):
    """No grammar: translate (complete) → compile → test → review APPROVED (generate)."""
    client = _mock_client(
        complete_responses=[_FULL],
        generate_responses=["APPROVED"],
    )
    result = generate_with_refinement(
        "add", PYTORCH_CODE, "assert True\n", client, max_iters=5
    )
    assert result.passed is True
    assert result.total_iterations == 1
    assert client.complete.call_count == 1  # translate only
    assert client.generate.call_count == 1  # review only


# ---------------------------------------------------------------------------
# Test 2: Grammar two-step — kernel + wrapper calls
# ---------------------------------------------------------------------------


@patch("core.loop._run_code", return_value=(True, ""))
def test_grammar_two_step(mock_run):
    """With grammar: kernel (complete, grammar) + wrapper (generate, chat) → review (generate, chat)."""
    client = _mock_client(
        complete_responses=[_KERNEL],
        generate_responses=[_WRAPPER, "APPROVED"],
    )
    result = generate_with_refinement(
        "add", PYTORCH_CODE, "assert True\n", client, grammar="fake", max_iters=5
    )
    assert result.passed is True
    assert "@triton.jit" in result.final_code
    assert "add_triton" in result.final_code
    # Kernel call uses complete with grammar
    first_call_kwargs = client.complete.call_args_list[0][1]
    assert first_call_kwargs.get("grammar") == "fake"
    # Wrapper uses generate (chat API)
    assert client.generate.call_count >= 1


# ---------------------------------------------------------------------------
# Test 3: Compile failure → fix → success
# ---------------------------------------------------------------------------


def test_compile_failure_then_fix():
    """Compile fails → fixer called (generate) → second run passes."""
    run_side_effects = [
        (False, "SyntaxError: invalid syntax"),  # compile iter 1
        (True, ""),                               # compile iter 2
        (True, ""),                               # test iter 2
    ]
    client = _mock_client(
        complete_responses=[_FULL],
        generate_responses=[_FIXED, "APPROVED"],
    )
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
    """Compile passes, test fails, fix called, second iteration passes."""
    run_side_effects = [
        (True, ""),                                # compile iter 1
        (False, "AssertionError: values differ"),  # test iter 1
        (True, ""),                                # compile iter 2
        (True, ""),                                # test iter 2
    ]
    client = _mock_client(
        complete_responses=[_FULL],
        generate_responses=[_FIXED, "APPROVED"],
    )
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
    """Compile ok, test ok, reviewer rejects, fix, retry approved."""
    run_side_effects = [
        (True, ""),  # compile iter 1
        (True, ""),  # test iter 1
        (True, ""),  # compile iter 2
        (True, ""),  # test iter 2
    ]
    client = _mock_client(
        complete_responses=[_FULL],
        generate_responses=[
            "Missing masks in tl.load",  # review rejects
            _FIXED,                       # fix
            "APPROVED",                   # review approves
        ],
    )
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
    fix1 = "```python\n@triton.jit\ndef kernel(x_ptr, BLOCK: tl.constexpr):\n    pass  # fix1\n\ndef add_triton(x, y):\n    kernel[(1,)](x)\n    return x\n```"
    fix2 = "```python\n@triton.jit\ndef kernel(x_ptr, BLOCK: tl.constexpr):\n    pass  # fix2\n\ndef add_triton(x, y):\n    kernel[(1,)](x)\n    return x\n```"
    client = _mock_client(
        complete_responses=[_FULL],
        generate_responses=[fix1, fix2],
    )
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
    client = _mock_client(
        complete_responses=[_FULL],
        generate_responses=["APPROVED"],
    )
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
    client = _mock_client(
        complete_responses=[_FULL],
        generate_responses=["APPROVED"],
    )
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
    client = _mock_client(
        complete_responses=[_FULL],
        generate_responses=["APPROVED"],
    )
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
    fix1 = "```python\n@triton.jit\ndef kernel(): pass  # fix1\n\ndef add_triton(x, y): return x\n```"
    client = _mock_client(
        complete_responses=[_FULL],
        generate_responses=[fix1],
    )
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


# ---------------------------------------------------------------------------
# Test: Multi-turn fixer accumulates context
# ---------------------------------------------------------------------------


def test_fixer_accumulates_messages():
    """After two fix calls, client.generate receives growing message lists."""
    run_side_effects = [
        (False, "SyntaxError: line 1"),   # compile iter 1 — fail
        (False, "NameError: undefined"),   # compile iter 2 — fail
        (True, ""),                        # compile iter 3 — pass
        (True, ""),                        # test iter 3 — pass
    ]
    fix1 = "```python\n@triton.jit\ndef kernel(): pass  # fix1\n\ndef add_triton(x, y): return x\n```"
    fix2 = "```python\n@triton.jit\ndef kernel(): pass  # fix2\n\ndef add_triton(x, y): return x\n```"
    client = _mock_client(
        complete_responses=[_FULL],
        generate_responses=[fix1, fix2, "APPROVED"],
    )
    with patch("core.loop._run_code", side_effect=run_side_effects):
        result = generate_with_refinement(
            "add", PYTORCH_CODE, "assert True\n", client, max_iters=5
        )

    assert result.passed is True
    # generate was called 3 times: fix1, fix2, review
    assert client.generate.call_count == 3

    # Both fix calls share the same fixer_messages list (mutated in place).
    # After the loop, the list has grown to:
    #   [system, user, assistant(fix1), user(followup), assistant(fix2)] → 5 items.
    # MagicMock stores a reference, so both call args point to the final list.
    first_fix_call = client.generate.call_args_list[0]
    first_msgs = first_fix_call[0][0]  # positional arg: messages (same list object)
    assert first_msgs[0]["role"] == "system"  # first message is always system
    assert first_msgs[1]["role"] == "user"    # second is the initial user prompt

    # The final fixer thread has accumulated all turns: 5 messages total
    assert len(first_msgs) == 5

    # Roles alternate correctly across the full thread
    roles = [m["role"] for m in first_msgs]
    assert roles == ["system", "user", "assistant", "user", "assistant"]

    # Both fix calls referenced the same growing list object
    second_fix_call = client.generate.call_args_list[1]
    assert second_fix_call[0][0] is first_msgs  # same object, confirms multi-turn sharing
