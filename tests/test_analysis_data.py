"""Tests for analysis/data.py -- experiment DataFrame builder."""

import json
import tempfile
from pathlib import Path

from analysis.data import build_experiment_df, _parse_run_dir_name


def test_parse_run_dir_name_refinement():
    conditions = ["single-shot", "refinement"]
    result = _parse_run_dir_name("Qwen2.5-Coder-7B_refinement_seed42", conditions)
    assert result == ("Qwen2.5-Coder-7B", "refinement", 42)


def test_parse_run_dir_name_single_shot():
    conditions = ["single-shot", "refinement"]
    result = _parse_run_dir_name("deepseek-coder-6.7b-base_single-shot_seed123", conditions)
    assert result == ("deepseek-coder-6.7b-base", "single-shot", 123)


def test_parse_run_dir_name_no_match():
    conditions = ["single-shot", "refinement"]
    result = _parse_run_dir_name("random_folder", conditions)
    assert result is None


def _write_run_dir(
    base: Path,
    model: str,
    condition: str,
    seed: int,
    ops: dict[str, dict],
) -> None:
    """Create a fake run directory with op_results.json and trajectories."""
    run_id = f"{model}_{condition}_seed{seed}"
    run_dir = base / run_id
    traj_dir = run_dir / "trajectories"
    traj_dir.mkdir(parents=True)

    (run_dir / "op_results.json").write_text(
        json.dumps(ops, indent=2), encoding="utf-8",
    )
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")

    for op_name in ops:
        traj_line = json.dumps({
            "op_name": op_name, "iteration": 1, "stage": "compile",
            "prompt_hash": "", "generation": "code",
            "outcome": "pass" if ops[op_name]["phase1_compile"] else "fail",
            "error": None, "timestamp": "t0",
        })
        (traj_dir / f"{op_name}.jsonl").write_text(traj_line + "\n", encoding="utf-8")


def test_build_experiment_df_from_op_results():
    """build_experiment_df reads op_results.json as primary source."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        ops = {
            "torch_add": {
                "phase1_compile": True,
                "phase2_test": True,
                "review_approved": True,
                "iterations_used": 1,
                "repaired": False,
                "phases_reached": ["compile", "test", "review"],
                "phases_skipped": [],
                "final_status": "pass",
            },
            "torch_bmm": {
                "phase1_compile": True,
                "phase2_test": False,
                "review_approved": None,
                "iterations_used": 5,
                "repaired": False,
                "phases_reached": ["compile", "test"],
                "phases_skipped": ["review"],
                "final_status": "fail",
            },
        }
        _write_run_dir(base, "TestModel", "refinement", 42, ops)
        _write_run_dir(base, "TestModel", "single-shot", 42, ops)

        df = build_experiment_df(base, ["single-shot", "refinement"])

    assert len(df) == 4  # 2 ops x 2 conditions
    assert set(df.columns) >= {"model", "condition", "seed", "op_name", "phase1", "phase2", "iterations", "repaired", "phases_skipped"}

    add_ref = df[(df["op_name"] == "torch_add") & (df["condition"] == "refinement")]
    assert add_ref.iloc[0]["phase1"] == 1
    assert add_ref.iloc[0]["phase2"] == 1

    bmm_ref = df[(df["op_name"] == "torch_bmm") & (df["condition"] == "refinement")]
    assert bmm_ref.iloc[0]["phase1"] == 1
    assert bmm_ref.iloc[0]["phase2"] == 0


def test_build_experiment_df_skipped_phase_is_nan():
    """When phase2_test is null (skipped), phase2 should be 0 and phases_skipped populated."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        ops = {
            "torch_svd": {
                "phase1_compile": False,
                "phase2_test": None,
                "review_approved": None,
                "iterations_used": 5,
                "repaired": False,
                "phases_reached": ["compile"],
                "phases_skipped": ["test", "review"],
                "final_status": "fail",
            },
        }
        _write_run_dir(base, "TestModel", "refinement", 42, ops)

        df = build_experiment_df(base, ["single-shot", "refinement"])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["phase1"] == 0
    assert row["phase2"] == 0  # skipped -> 0
    assert "test" in row["phases_skipped"]
