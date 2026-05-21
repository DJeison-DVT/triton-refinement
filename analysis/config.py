"""Analysis configuration -- loaded from experiment_config.json.

Dataclasses and loader extracted from scripts/analyze.py so that every
analysis module can share the same typed config without globals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    """A pairwise comparison between two experimental conditions."""

    name: str
    condition_a: str
    condition_b: str


@dataclass
class AnalysisConfig:
    """Configuration for the analysis pipeline.

    Loaded from experiment_config.json. Replaces hardcoded constants so
    comparisons can be swapped in/out for ablation studies.
    """

    models: list[str]
    seeds: list[int]
    conditions: list[str]
    comparisons: list[Comparison]
    steps: list[str]
    alpha: float = 0.05
    n_ops: int = 166


# Steps that can be enabled/disabled via config
DEFAULT_STEPS = [
    "power_analysis",
    "pairwise",
    "friedman",
    "holm_correction",
    "cross_seed_aggregation",
]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> AnalysisConfig:
    """Load analysis config from experiment_config.json.

    Reads models, conditions, seeds, and statistical params from the
    experiment config. If an 'analysis' section is present, it provides
    custom comparisons and step selection. Otherwise, sensible defaults
    are generated (each condition compared to the first condition).
    """
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    models = [m["name"] for m in raw["models"]]
    seeds = raw["reproducibility"]["seeds"]
    alpha = raw["statistical_plan"]["alpha"]
    n_ops = raw["benchmark"]["n_ops"]
    conditions = list(raw["conditions"].keys())

    # Custom analysis section, or derive defaults
    analysis = raw.get("analysis", {})
    steps = analysis.get("steps", DEFAULT_STEPS)

    if "comparisons" in analysis:
        comparisons = [
            Comparison(c["name"], c["condition_a"], c["condition_b"])
            for c in analysis["comparisons"]
        ]
    else:
        # Default: compare every condition to the first one (baseline)
        baseline = conditions[0]
        comparisons = [
            Comparison(
                name=f"{baseline}_vs_{cond}",
                condition_a=baseline,
                condition_b=cond,
            )
            for cond in conditions[1:]
        ]

    return AnalysisConfig(
        models=models,
        seeds=seeds,
        conditions=conditions,
        comparisons=comparisons,
        steps=steps,
        alpha=alpha,
        n_ops=n_ops,
    )
