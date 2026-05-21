"""Trajectory-only analysis for local development runs (week 2).

Extracts metrics from raw JSONL trajectory files without requiring
TritonBench4Modal evaluation results. Works with the flat layout:
    results/local/trajectories/{op_name}.jsonl

Metrics computed:
    - Internal compile pass rate per iteration
    - Internal test pass rate per iteration
    - Repair success rate (failed initially → passed later)
    - Iterations to first test pass
    - Error category breakdown
    - Per-op summary (final status, iterations used)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_trajectories(traj_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load all JSONL trajectory files from a directory.

    Returns dict mapping op_name → list of trajectory records.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(traj_dir.glob("*.jsonl")):
        records = []
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        if records:
            op_name = records[0].get("op_name", f.stem)
            result[op_name] = records
    return result


def summarize_op(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a single op's trajectory.

    Returns dict with:
        - max_iteration: highest iteration number
        - final_compile: pass/fail of last compile stage
        - final_test: pass/fail of last test stage (or None if never reached)
        - final_review: pass/fail of last review stage (or None if never reached)
        - passed: True if review approved or tests passed on final iteration
        - iterations_to_first_test_pass: iteration number, or None
        - repaired: True if compile/test failed on iter 1 but passed later
        - error_types: list of error category strings
    """
    max_iter = max((r["iteration"] for r in records), default=0)

    # Collect per-stage outcomes
    compiles = [(r["iteration"], r["outcome"]) for r in records if r["stage"] == "compile"]
    tests = [(r["iteration"], r["outcome"]) for r in records if r["stage"] == "test"]
    reviews = [(r["iteration"], r["outcome"]) for r in records if r["stage"] == "review"]

    final_compile = compiles[-1][1] if compiles else None
    final_test = tests[-1][1] if tests else None
    final_review = reviews[-1][1] if reviews else None

    # Iterations to first test pass
    first_test_pass = None
    for it, outcome in tests:
        if outcome == "pass":
            first_test_pass = it
            break

    # Repair: first compile or test failed, but later passed
    first_compile_failed = compiles[0][1] == "fail" if compiles else False
    first_test_failed = tests[0][1] == "fail" if tests else False
    later_test_passed = any(o == "pass" for _, o in tests[1:]) if len(tests) > 1 else False
    later_compile_passed = any(o == "pass" for _, o in compiles[1:]) if len(compiles) > 1 else False
    repaired = (first_compile_failed and later_compile_passed) or (first_test_failed and later_test_passed)

    # Overall pass: review approved or last test passed
    passed = final_review == "pass" or final_test == "pass"

    # Error categories
    error_types = []
    for r in records:
        if r.get("error"):
            error_types.append(_categorize_error(r["error"]))

    return {
        "max_iteration": max_iter,
        "final_compile": final_compile,
        "final_test": final_test,
        "final_review": final_review,
        "passed": passed,
        "iterations_to_first_test_pass": first_test_pass,
        "repaired": repaired,
        "error_types": error_types,
    }


def _categorize_error(error: str) -> str:
    """Categorize an error string into a high-level category."""
    if "SyntaxError" in error:
        return "SyntaxError"
    if "NameError" in error:
        return "NameError"
    if "ModuleNotFoundError" in error:
        return "ModuleNotFoundError"
    if "ImportError" in error:
        return "ImportError"
    if "TypeError" in error:
        return "TypeError"
    if "AttributeError" in error:
        return "AttributeError"
    if "CompilationError" in error or "triton.compiler" in error:
        return "TritonCompilationError"
    if "TimeoutExpired" in error:
        return "Timeout"
    if "AssertionError" in error:
        return "AssertionError"
    # Try to extract the last exception class
    match = re.search(r"(\w+Error|\w+Exception)", error)
    if match:
        return match.group(1)
    return "Unknown"


def analyze_trajectories(traj_dir: Path) -> dict[str, Any]:
    """Run full trajectory analysis on a directory of JSONL files.

    Returns a dict with:
        - n_ops: number of operations analyzed
        - per_op: dict of op_name → summarize_op result
        - compile_pass_rate: fraction of ops whose final compile passed
        - test_pass_rate: fraction of ops whose final test passed
        - overall_pass_rate: fraction of ops that passed (review approved or test passed)
        - repair_rate: fraction of ops that were repaired
        - mean_iterations: mean iterations used across all ops
        - error_counts: Counter of error categories
        - per_iteration_compile: list of (iteration, compile_pass_rate)
        - per_iteration_test: list of (iteration, test_pass_rate)
    """
    all_trajs = load_trajectories(traj_dir)
    if not all_trajs:
        return {"n_ops": 0, "error": "no trajectory files found"}

    per_op = {name: summarize_op(records) for name, records in all_trajs.items()}
    n_ops = len(per_op)

    # Aggregate rates
    compile_passed = sum(1 for s in per_op.values() if s["final_compile"] == "pass")
    test_passed = sum(1 for s in per_op.values() if s["final_test"] == "pass")
    overall_passed = sum(1 for s in per_op.values() if s["passed"])
    repaired = sum(1 for s in per_op.values() if s["repaired"])
    iterations = [s["max_iteration"] for s in per_op.values()]

    # Error breakdown
    error_counts: Counter[str] = Counter()
    for s in per_op.values():
        error_counts.update(s["error_types"])

    # Per-iteration rates
    max_iter = max(iterations) if iterations else 0
    all_records = [r for records in all_trajs.values() for r in records]

    per_iter_compile = []
    per_iter_test = []
    for i in range(1, max_iter + 1):
        compile_at_i = [r for r in all_records if r["iteration"] == i and r["stage"] == "compile"]
        test_at_i = [r for r in all_records if r["iteration"] == i and r["stage"] == "test"]
        if compile_at_i:
            rate = sum(1 for r in compile_at_i if r["outcome"] == "pass") / len(compile_at_i)
            per_iter_compile.append((i, rate))
        if test_at_i:
            rate = sum(1 for r in test_at_i if r["outcome"] == "pass") / len(test_at_i)
            per_iter_test.append((i, rate))

    return {
        "n_ops": n_ops,
        "per_op": per_op,
        "compile_pass_rate": compile_passed / n_ops,
        "test_pass_rate": test_passed / n_ops,
        "overall_pass_rate": overall_passed / n_ops,
        "repair_rate": repaired / n_ops,
        "repair_count": repaired,
        "mean_iterations": sum(iterations) / len(iterations) if iterations else 0,
        "error_counts": dict(error_counts.most_common()),
        "per_iteration_compile": per_iter_compile,
        "per_iteration_test": per_iter_test,
    }


def print_trajectory_report(results: dict[str, Any]) -> None:
    """Print a human-readable trajectory analysis report."""
    if results.get("error"):
        print(f"Error: {results['error']}")
        return

    n = results["n_ops"]
    print(f"\n=== Trajectory Analysis ({n} ops) ===\n")

    print(f"  Compile pass rate (final):  {results['compile_pass_rate']:.1%}")
    print(f"  Test pass rate (final):     {results['test_pass_rate']:.1%}")
    print(f"  Overall pass rate:          {results['overall_pass_rate']:.1%}")
    print(f"  Repair rate:                {results['repair_count']}/{n} ({results['repair_rate']:.1%})")
    print(f"  Mean iterations used:       {results['mean_iterations']:.1f}")

    if results["per_iteration_compile"]:
        print(f"\n  --- Compile pass rate by iteration ---")
        for i, rate in results["per_iteration_compile"]:
            bar = "#" * int(rate * 20)
            print(f"    iter {i}: {rate:.1%}  {bar}")

    if results["per_iteration_test"]:
        print(f"\n  --- Test pass rate by iteration ---")
        for i, rate in results["per_iteration_test"]:
            bar = "#" * int(rate * 20)
            print(f"    iter {i}: {rate:.1%}  {bar}")

    if results["error_counts"]:
        print(f"\n  --- Error categories ---")
        for cat, count in results["error_counts"].items():
            print(f"    {cat}: {count}")

    print(f"\n  --- Per-op results ---")
    for op_name, s in results["per_op"].items():
        status = "PASS" if s["passed"] else "FAIL"
        repair_tag = " [repaired]" if s["repaired"] else ""
        iters = s["max_iteration"]
        first_pass = s["iterations_to_first_test_pass"]
        first_pass_str = f", first test pass @ iter {first_pass}" if first_pass else ""
        print(f"    {op_name}: {status} ({iters} iters{first_pass_str}){repair_tag}")
