"""Data loading utilities for Triton Refinement analysis.

Functions extracted from scripts/analyze.py to provide a reusable,
importable module. The key design change from the original: functions that
previously relied on a global CONDITIONS list now accept `conditions` as
an explicit parameter, supplied by the caller from AnalysisConfig.conditions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL trajectory file."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_evaluation_results(path: Path) -> dict[str, Any]:
    """Load TritonBench4Modal evaluation output (predictions + phases)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_experiment_df(results_dir: Path, conditions: list[str]) -> pd.DataFrame:
    """Scan results directory and build a unified DataFrame.

    Reads op_results.json as the primary data source for per-op outcomes.
    Falls back to trajectory parsing + evaluation files when op_results.json
    is missing (backward compatibility with pre-week3 runs).

    Expected layout::

        results_dir/
            {model}_{condition}_seed{seed}/
                op_results.json              # primary: per-op outcomes (week 3+)
                trajectories/
                    {op_name}.jsonl
                evaluation.json              # optional: TritonBench4Modal output
    """
    rows: list[dict[str, Any]] = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        meta = _parse_run_dir_name(run_dir.name, conditions)
        if meta is None:
            continue

        model, condition, seed = meta

        # Try op_results.json first (week 3+ format)
        op_results_file = run_dir / "op_results.json"
        if op_results_file.exists():
            op_results = json.loads(op_results_file.read_text(encoding="utf-8"))
            eval_data = _load_eval_data(run_dir)

            for op_name, op in op_results.items():
                ev = eval_data.get(op_name, {})
                rows.append({
                    "model": model,
                    "condition": condition,
                    "seed": seed,
                    "op_name": op_name,
                    "phase1": int(
                        op.get("eval_phase1",
                               ev.get("phase1",
                                      op.get("phase1_compile", False) or False))
                    ),
                    "phase2": int(
                        op.get("eval_phase2",
                               ev.get("phase2",
                                      op.get("phase2_test", False) or False))
                    ),
                    "speedup": op.get("eval_speedup", ev.get("speedup", np.nan)),
                    "runtime": ev.get("runtime", np.nan),
                    "iterations": op.get("iterations_used", 1),
                    "repaired": int(op.get("repaired", False)),
                    "phases_skipped": op.get("phases_skipped", []),
                })
            continue

        # Fallback: trajectory parsing (pre-week3 runs)
        traj_dir = run_dir / "trajectories"
        if not traj_dir.exists():
            traj_dir = run_dir

        eval_data = _load_eval_data(run_dir)

        op_files = sorted(traj_dir.glob("*.jsonl"))
        for op_file in op_files:
            op_name = op_file.stem
            traj = load_trajectory(op_file)

            iterations = _count_iterations(traj)
            first_outcome = _first_iteration_outcome(traj)
            final_outcome = _final_outcome(traj)

            phase1 = eval_data.get(op_name, {}).get("phase1", final_outcome == "pass")
            phase2 = eval_data.get(op_name, {}).get("phase2", False)
            speedup = eval_data.get(op_name, {}).get("speedup", np.nan)
            runtime = eval_data.get(op_name, {}).get("runtime", np.nan)
            repaired = (first_outcome == "fail") and (final_outcome == "pass")

            rows.append({
                "model": model,
                "condition": condition,
                "seed": seed,
                "op_name": op_name,
                "phase1": int(phase1),
                "phase2": int(phase2),
                "speedup": speedup,
                "runtime": runtime,
                "iterations": iterations,
                "repaired": int(repaired),
                "phases_skipped": [],
            })

    if not rows:
        print(
            "WARNING: No experiment data found. Generating empty DataFrame.",
            file=sys.stderr,
        )
        return pd.DataFrame(columns=[
            "model", "condition", "seed", "op_name",
            "phase1", "phase2", "speedup", "runtime",
            "iterations", "repaired", "phases_skipped",
        ])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_run_dir_name(
    name: str, conditions: list[str]
) -> tuple[str, str, int] | None:
    """Parse ``'{model}_{condition}_seed{seed}'`` directory name.

    Parameters
    ----------
    name:
        Bare directory name (not a full path).
    conditions:
        Ordered list of condition strings to match against (e.g.
        ``["baseline", "reflexion"]``).

    Returns
    -------
    tuple[str, str, int] | None
        ``(model, condition, seed)`` on success, ``None`` if the name does
        not match any known condition.
    """
    for condition in conditions:
        if f"_{condition}_seed" in name:
            prefix, _, seed_str = name.rpartition("_seed")
            model = prefix.rsplit(f"_{condition}", 1)[0]
            try:
                seed = int(seed_str)
            except ValueError:
                return None
            return model, condition, seed
    return None


def _load_eval_data(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load evaluation data from a run directory.

    Tries ``evaluation.json`` first; falls back to individual
    ``phase{1,2,3}_results.json`` files.
    """
    result: dict[str, dict[str, Any]] = {}

    eval_file = run_dir / "evaluation.json"
    if eval_file.exists():
        data = json.loads(eval_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for op_name, metrics in data.items():
                if isinstance(metrics, dict):
                    result[op_name] = metrics
        return result

    # Try individual phase files
    for phase_name in ["phase1_results", "phase2_results", "phase3_results"]:
        phase_file = run_dir / f"{phase_name}.json"
        if phase_file.exists():
            data = json.loads(phase_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for op_name, val in data.items():
                    if op_name not in result:
                        result[op_name] = {}
                    if "phase1" in phase_name:
                        result[op_name]["phase1"] = bool(val)
                    elif "phase2" in phase_name:
                        result[op_name]["phase2"] = bool(val)
                    elif "phase3" in phase_name:
                        result[op_name]["speedup"] = (
                            float(val) if val is not None else np.nan
                        )

    return result


def _count_iterations(traj: list[dict]) -> int:
    """Count refinement iterations from trajectory."""
    if not traj:
        return 0
    iters = {r.get("iteration", 0) for r in traj}
    return max(iters) + 1 if iters else 1


def _first_iteration_outcome(traj: list[dict]) -> str:
    """Get outcome of the first iteration."""
    first_iter = [r for r in traj if r.get("iteration", 0) == 0]
    for r in reversed(first_iter):
        if r.get("outcome") in ("pass", "fail"):
            return r["outcome"]
    return "fail"


def _final_outcome(traj: list[dict]) -> str:
    """Get outcome of the final iteration."""
    for r in reversed(traj):
        if r.get("outcome") in ("pass", "fail"):
            return r["outcome"]
    return "fail"
