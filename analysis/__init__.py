"""Statistical analysis package for Triton Refinement experiments.

Submodules:
    config   — AnalysisConfig dataclass and experiment_config.json loader
    data     — trajectory/evaluation data loading into pandas DataFrames
    stats    — statistical tests, effect sizes, CIs, power analysis, correction
    pipeline — pairwise analysis, multi-model comparison, seed aggregation
    output   — LaTeX table generators and matplotlib plots
"""

from analysis.config import AnalysisConfig, Comparison, load_config
from analysis.data import build_experiment_df
from analysis.pipeline import (
    aggregate_across_seeds,
    apply_holm_correction,
    run_multi_model_comparison,
    run_pairwise_analysis,
)

__all__ = [
    "AnalysisConfig",
    "Comparison",
    "load_config",
    "build_experiment_df",
    "run_pairwise_analysis",
    "run_multi_model_comparison",
    "aggregate_across_seeds",
    "apply_holm_correction",
]
