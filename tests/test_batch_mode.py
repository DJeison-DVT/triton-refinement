"""Tests for run_experiment.py batch mode and condition handling."""

import json
import tempfile
from pathlib import Path


def _make_config(tmp: Path) -> Path:
    """Write a minimal experiment_config.json and return its path."""
    config = {
        "models": [
            {"name": "TestModel-7B", "hf_id": "test/TestModel-7B", "params": "7B", "role": "headline"},
        ],
        "conditions": {
            "single-shot": {"max_iterations": 1, "grammar_constrained": True, "pattern_memory": False},
            "refinement": {"max_iterations": 5, "grammar_constrained": True, "pattern_memory": "inmemory"},
        },
        "reproducibility": {"seeds": [42, 123], "temperature": 0.0, "deterministic_seeds": {"random": True, "numpy": True, "torch": True}},
        "benchmark": {"name": "TritonBench-T", "n_ops": 166, "data_source": "thunlp/TritonBench", "evaluator": "TritonBench4Modal", "evaluator_commit": "6103f69"},
        "grammar": {"source": "triton_lexeme/triton.ebnf", "backend": "xgrammar", "api_param": "guided_grammar"},
        "statistical_plan": {"alpha": 0.05, "power": 0.80, "primary_metric": "phase2_pass_rate", "tests": {}, "effect_sizes": {}},
    }
    p = tmp / "experiment_config.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    return p


def test_make_run_id():
    from scripts.run_experiment import make_run_id
    assert make_run_id("Qwen/Qwen2.5-Coder-7B", "refinement", 42) == "Qwen2.5-Coder-7B_refinement_seed42"
    assert make_run_id("deepseek-ai/deepseek-coder-6.7b-base", "single-shot", 123) == "deepseek-coder-6.7b-base_single-shot_seed123"


def test_build_sweep_matrix_full():
    from scripts.run_experiment import build_sweep_matrix
    with tempfile.TemporaryDirectory() as tmp:
        config_path = _make_config(Path(tmp))
        matrix = build_sweep_matrix(config_path)
    # 1 model x 2 conditions x 2 seeds = 4 runs
    assert len(matrix) == 4
    assert all("model" in r and "condition" in r and "seed" in r for r in matrix)


def test_build_sweep_matrix_filtered():
    from scripts.run_experiment import build_sweep_matrix
    with tempfile.TemporaryDirectory() as tmp:
        config_path = _make_config(Path(tmp))
        matrix = build_sweep_matrix(config_path, conditions_filter=["single-shot"])
    # 1 model x 1 condition x 2 seeds = 2 runs
    assert len(matrix) == 2
    assert all(r["condition"] == "single-shot" for r in matrix)


def test_is_run_complete_false_when_missing(tmp_path):
    from scripts.run_experiment import is_run_complete
    assert is_run_complete(tmp_path / "nonexistent") is False


def test_is_run_complete_true_when_summary_exists(tmp_path):
    from scripts.run_experiment import is_run_complete
    run_dir = tmp_path / "some_run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    assert is_run_complete(run_dir) is True


def test_resolve_condition_settings():
    from scripts.run_experiment import resolve_condition_settings
    with tempfile.TemporaryDirectory() as tmp:
        config_path = _make_config(Path(tmp))
        settings = resolve_condition_settings(config_path, "refinement")
    assert settings["max_iterations"] == 5
    assert settings["grammar_constrained"] is True
    assert settings["pattern_memory"] == "inmemory"
