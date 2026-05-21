"""Tests for build_op_result() — per-op outcome extraction."""

from core.loop import (
    IterationLog,
    RefinementResult,
    build_op_result,
)


PYTORCH_CODE = "def add(x, y):\n    return x + y\n"


def _make_result(trajectory: list[IterationLog], passed: bool, total_iterations: int) -> RefinementResult:
    return RefinementResult(
        op_name="torch_add",
        final_code="@triton.jit\ndef k(): pass\n",
        test_code="assert True\n",
        passed=passed,
        total_iterations=total_iterations,
        trajectory=trajectory,
    )


def test_happy_path_all_phases_reached():
    """Compile pass, test pass, review approved -> all phases reached."""
    traj = [
        IterationLog(0, "translate", "h1", "code", "pass", None, "t0"),
        IterationLog(1, "compile", "", "code", "pass", None, "t1"),
        IterationLog(1, "test", "", "code", "pass", None, "t2"),
        IterationLog(1, "review", "h2", "APPROVED", "pass", None, "t3"),
    ]
    result = _make_result(traj, passed=True, total_iterations=1)
    op = build_op_result(result)

    assert op["phase1_compile"] is True
    assert op["phase2_test"] is True
    assert op["review_approved"] is True
    assert op["iterations_used"] == 1
    assert op["repaired"] is False
    assert op["phases_reached"] == ["compile", "test", "review"]
    assert op["phases_skipped"] == []
    assert op["final_status"] == "pass"


def test_compile_fail_skips_test_and_review():
    """Compile always fails -> test and review are skipped (null)."""
    traj = [
        IterationLog(0, "translate", "h1", "code", "pass", None, "t0"),
        IterationLog(1, "compile", "", "code", "fail", "SyntaxError", "t1"),
        IterationLog(1, "fix", "h2", "code", "pass", None, "t2"),
        IterationLog(2, "compile", "", "code", "fail", "SyntaxError", "t3"),
        IterationLog(2, "fix", "h3", "code", "pass", None, "t4"),
    ]
    result = _make_result(traj, passed=False, total_iterations=2)
    op = build_op_result(result)

    assert op["phase1_compile"] is False
    assert op["phase2_test"] is None
    assert op["review_approved"] is None
    assert op["phases_reached"] == ["compile"]
    assert op["phases_skipped"] == ["test", "review"]
    assert op["final_status"] == "fail"


def test_test_fail_skips_review():
    """Compile passes, test fails -> review skipped."""
    traj = [
        IterationLog(0, "translate", "h1", "code", "pass", None, "t0"),
        IterationLog(1, "compile", "", "code", "pass", None, "t1"),
        IterationLog(1, "test", "", "code", "fail", "AssertionError", "t2"),
        IterationLog(1, "fix", "h2", "code", "pass", None, "t3"),
        IterationLog(2, "compile", "", "code", "pass", None, "t4"),
        IterationLog(2, "test", "", "code", "fail", "AssertionError", "t5"),
    ]
    result = _make_result(traj, passed=False, total_iterations=2)
    op = build_op_result(result)

    assert op["phase1_compile"] is True
    assert op["phase2_test"] is False
    assert op["review_approved"] is None
    assert op["phases_reached"] == ["compile", "test"]
    assert op["phases_skipped"] == ["review"]


def test_repaired_op():
    """Compile fails iter 1, passes iter 2 with test+review -> repaired=True."""
    traj = [
        IterationLog(0, "translate", "h1", "code", "pass", None, "t0"),
        IterationLog(1, "compile", "", "code", "fail", "SyntaxError", "t1"),
        IterationLog(1, "fix", "h2", "code", "pass", None, "t2"),
        IterationLog(2, "compile", "", "code", "pass", None, "t3"),
        IterationLog(2, "test", "", "code", "pass", None, "t4"),
        IterationLog(2, "review", "h3", "APPROVED", "pass", None, "t5"),
    ]
    result = _make_result(traj, passed=True, total_iterations=2)
    op = build_op_result(result)

    assert op["repaired"] is True
    assert op["phase1_compile"] is True
    assert op["phase2_test"] is True
    assert op["review_approved"] is True
    assert op["iterations_used"] == 2


def test_review_rejected_never_approved():
    """Compile+test pass, review rejects, fix, compile+test pass, review rejects again."""
    traj = [
        IterationLog(0, "translate", "h1", "code", "pass", None, "t0"),
        IterationLog(1, "compile", "", "code", "pass", None, "t1"),
        IterationLog(1, "test", "", "code", "pass", None, "t2"),
        IterationLog(1, "review", "h2", "Missing masks", "fail", None, "t3"),
        IterationLog(1, "fix", "h3", "code", "pass", None, "t4"),
        IterationLog(2, "compile", "", "code", "pass", None, "t5"),
        IterationLog(2, "test", "", "code", "pass", None, "t6"),
        IterationLog(2, "review", "h4", "Still missing", "fail", None, "t7"),
    ]
    result = _make_result(traj, passed=False, total_iterations=2)
    op = build_op_result(result)

    assert op["phase1_compile"] is True
    assert op["phase2_test"] is True
    assert op["review_approved"] is False
    assert op["phases_reached"] == ["compile", "test", "review"]
    assert op["phases_skipped"] == []
