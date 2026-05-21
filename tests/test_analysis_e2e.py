"""End-to-end test: full analysis pipeline on synthetic op_results.json data."""

import json
import tempfile
from pathlib import Path

import numpy as np

from analysis.config import AnalysisConfig, Comparison
from analysis.data import build_experiment_df
from analysis.pipeline import (
    run_pairwise_analysis,
    run_multi_model_comparison,
    aggregate_across_seeds,
    apply_holm_correction,
)
from analysis.output import results_to_latex_main_table, results_to_latex_repair_table


def _make_ops(n: int, pass_rate_p1: float, pass_rate_p2: float, seed: int = 0) -> dict:
    """Generate n synthetic op results with given pass rates."""
    rng = np.random.RandomState(seed)
    ops = {}
    for i in range(n):
        p1 = bool(rng.random() < pass_rate_p1)
        p2 = bool(p1 and rng.random() < pass_rate_p2)
        phases_reached = ["compile"]
        phases_skipped = []
        if p1:
            phases_reached.append("test")
            if p2:
                phases_reached.append("review")
            else:
                phases_skipped.append("review")
        else:
            phases_skipped.extend(["test", "review"])

        ops[f"op_{i:03d}"] = {
            "phase1_compile": p1,
            "phase2_test": p2 if p1 else None,
            "review_approved": p2 if p2 else None,
            "iterations_used": 1 if p1 and p2 else 3,
            "repaired": False,
            "phases_reached": phases_reached,
            "phases_skipped": phases_skipped,
            "final_status": "pass" if p2 else "fail",
        }
    return ops


def _build_experiment(base: Path, n_ops: int = 30):
    """Create a minimal 2-model x 2-condition x 2-seed experiment."""
    models = ["ModelA", "ModelB"]
    conditions = ["single-shot", "refinement"]
    seeds = [42, 123]

    for model in models:
        for condition in conditions:
            for seed in seeds:
                run_id = f"{model}_{condition}_seed{seed}"
                run_dir = base / run_id
                traj_dir = run_dir / "trajectories"
                traj_dir.mkdir(parents=True)

                if condition == "single-shot":
                    ops = _make_ops(n_ops, pass_rate_p1=0.5, pass_rate_p2=0.3, seed=seed)
                else:
                    ops = _make_ops(n_ops, pass_rate_p1=0.8, pass_rate_p2=0.6, seed=seed)

                (run_dir / "op_results.json").write_text(json.dumps(ops, indent=2), encoding="utf-8")
                (run_dir / "summary.json").write_text("{}", encoding="utf-8")

                for op_name in ops:
                    traj_line = json.dumps({
                        "op_name": op_name, "iteration": 1, "stage": "compile",
                        "prompt_hash": "", "generation": "code",
                        "outcome": "pass" if ops[op_name]["phase1_compile"] else "fail",
                        "error": None, "timestamp": "t0",
                    })
                    (traj_dir / f"{op_name}.jsonl").write_text(traj_line + "\n", encoding="utf-8")


def _make_config() -> AnalysisConfig:
    return AnalysisConfig(
        models=["ModelA", "ModelB"],
        seeds=[42, 123],
        conditions=["single-shot", "refinement"],
        comparisons=[Comparison("ss_vs_ref", "single-shot", "refinement")],
        steps=["power_analysis", "pairwise", "friedman", "holm_correction", "cross_seed_aggregation"],
        alpha=0.05,
        n_ops=30,
    )


def test_build_df_correct_shape():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _build_experiment(base, n_ops=30)
        df = build_experiment_df(base, ["single-shot", "refinement"])
    # 2 models x 2 conditions x 2 seeds x 30 ops = 240
    assert len(df) == 240
    assert df["model"].nunique() == 2
    assert df["condition"].nunique() == 2
    assert df["seed"].nunique() == 2


def test_pairwise_analysis_runs():
    config = _make_config()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _build_experiment(base, n_ops=30)
        df = build_experiment_df(base, config.conditions)
        result = run_pairwise_analysis(df, "ModelA", config)

    assert "error" not in result
    assert "phase1" in result
    assert "phase2" in result
    assert "repair" in result
    assert "iterations" in result
    assert result["phase1"]["rate_refinement"] >= result["phase1"]["rate_single_shot"]


def test_friedman_runs():
    config = _make_config()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _build_experiment(base, n_ops=30)
        df = build_experiment_df(base, config.conditions)
        result = run_multi_model_comparison(df, config, "phase2")

    assert "friedman" in result or "error" in result


def test_holm_correction_runs():
    config = _make_config()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _build_experiment(base, n_ops=30)
        df = build_experiment_df(base, config.conditions)

        all_results = {}
        for model in config.models:
            all_results[model] = run_pairwise_analysis(df, model, config)

        corrections = apply_holm_correction(all_results, config.alpha)

    assert len(corrections) > 0
    for label, vals in corrections.items():
        assert "raw_p" in vals
        assert "corrected_p" in vals
        assert vals["corrected_p"] >= vals["raw_p"]


def test_cross_seed_aggregation_runs():
    config = _make_config()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _build_experiment(base, n_ops=30)
        df = build_experiment_df(base, config.conditions)
        summary = aggregate_across_seeds(df, config)

    assert "ModelA" in summary
    assert "phase1" in summary["ModelA"]
    assert "+/-" in summary["ModelA"]["phase1"]["rate_single_shot"]


def test_latex_tables_generate():
    config = _make_config()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _build_experiment(base, n_ops=30)
        df = build_experiment_df(base, config.conditions)

        all_results = {}
        for model in config.models:
            all_results[model] = run_pairwise_analysis(df, model, config)

    main_table = results_to_latex_main_table(all_results, config.alpha)
    assert r"\begin{table}" in main_table
    assert "ModelA" in main_table

    repair_table = results_to_latex_repair_table(all_results)
    assert r"\begin{table}" in repair_table
