"""Analysis orchestration for Triton Refinement experiments.

Extracted from scripts/analyze.py. All functions take explicit config
instead of global constants (ALPHA, MODELS, CONDITIONS).

Functions:
    run_pairwise_analysis   — single-shot vs refinement for one model
    run_multi_model_comparison — Friedman test across all models
    aggregate_across_seeds  — run per-seed, then mean +/- SD
    apply_holm_correction   — collect p-values and apply Holm-Bonferroni
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from analysis.config import AnalysisConfig
from analysis.stats import (
    bootstrap_ci_diff,
    cohens_d,
    cohens_h,
    friedman_test,
    holm_bonferroni,
    interpret_effect_size,
    mcnemar_test,
    wilcoxon_test,
    wilson_ci,
)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def run_pairwise_analysis(
    df: pd.DataFrame,
    model: str,
    config: AnalysisConfig,
    seed: int | None = None,
    condition_a: str | None = None,
    condition_b: str | None = None,
) -> dict[str, Any]:
    """Run the full pairwise analysis for one model comparing two conditions.

    If condition_a/condition_b are None, uses the first comparison from config.
    If seed is None, aggregates across all seeds for that model.
    Uses config.alpha for confidence intervals.
    """
    if condition_a is None or condition_b is None:
        if config.comparisons:
            condition_a = config.comparisons[0].condition_a
            condition_b = config.comparisons[0].condition_b
        else:
            condition_a = config.conditions[0]
            condition_b = config.conditions[1] if len(config.conditions) > 1 else config.conditions[0]

    if seed is not None:
        sub = df[(df["model"] == model) & (df["seed"] == seed)]
    else:
        sub = df[df["model"] == model]

    ss = sub[sub["condition"] == condition_a].set_index("op_name")
    ref = sub[sub["condition"] == condition_b].set_index("op_name")

    # Align by op_name
    common_ops = sorted(set(ss.index) & set(ref.index))
    if not common_ops:
        return {"error": "no common ops"}

    ss = ss.loc[common_ops]
    ref = ref.loc[common_ops]

    results: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "n_ops": len(common_ops),
    }

    # --- Phase 1 ---
    p1_ss = ss["phase1"].values
    p1_ref = ref["phase1"].values
    rate_ss_p1 = float(np.mean(p1_ss))
    rate_ref_p1 = float(np.mean(p1_ref))
    results["phase1"] = {
        "rate_single_shot": rate_ss_p1,
        "rate_refinement": rate_ref_p1,
        "delta": rate_ref_p1 - rate_ss_p1,
        "ci_single_shot": wilson_ci(int(np.sum(p1_ss)), len(p1_ss), config.alpha),
        "ci_refinement": wilson_ci(int(np.sum(p1_ref)), len(p1_ref), config.alpha),
        "mcnemar": mcnemar_test(p1_ss, p1_ref),
        "cohens_h": cohens_h(rate_ref_p1, rate_ss_p1),
    }

    # --- Phase 2 (primary metric) ---
    p2_ss = ss["phase2"].values
    p2_ref = ref["phase2"].values
    rate_ss_p2 = float(np.mean(p2_ss))
    rate_ref_p2 = float(np.mean(p2_ref))
    results["phase2"] = {
        "rate_single_shot": rate_ss_p2,
        "rate_refinement": rate_ref_p2,
        "delta": rate_ref_p2 - rate_ss_p2,
        "ci_single_shot": wilson_ci(int(np.sum(p2_ss)), len(p2_ss), config.alpha),
        "ci_refinement": wilson_ci(int(np.sum(p2_ref)), len(p2_ref), config.alpha),
        "mcnemar": mcnemar_test(p2_ss, p2_ref),
        "cohens_h": cohens_h(rate_ref_p2, rate_ss_p2),
    }

    # --- Runtime (H1: refinement < single-shot, one-sided "less") ---
    rt_ss = ss["runtime"].values.astype(float)
    rt_ref = ref["runtime"].values.astype(float)
    results["runtime"] = {
        "mean_single_shot": float(np.nanmean(rt_ss)),
        "mean_refinement": float(np.nanmean(rt_ref)),
        "wilcoxon": wilcoxon_test(rt_ss, rt_ref, alternative="less"),
        "cohens_d": cohens_d(rt_ss, rt_ref),
        "ci_diff": bootstrap_ci_diff(rt_ss, rt_ref, alpha=config.alpha),
    }

    # --- Speedup (H1: refinement > single-shot, one-sided "greater") ---
    sp_ss = ss["speedup"].values.astype(float)
    sp_ref = ref["speedup"].values.astype(float)
    results["speedup"] = {
        "geomean_single_shot": float(np.exp(np.nanmean(np.log(sp_ss[sp_ss > 0])))) if np.any(sp_ss > 0) else np.nan,
        "geomean_refinement": float(np.exp(np.nanmean(np.log(sp_ref[sp_ref > 0])))) if np.any(sp_ref > 0) else np.nan,
        "wilcoxon": wilcoxon_test(sp_ss, sp_ref, alternative="greater"),
        "cohens_d": cohens_d(sp_ss, sp_ref),
        "ci_diff": bootstrap_ci_diff(sp_ss, sp_ref, alpha=config.alpha),
    }

    # --- Repair success (H0: p_repair = 0, H1: p_repair > 0) ---
    repair_count = int(ref["repaired"].sum())
    failed_initially = int(np.sum(p1_ss == 0))  # ops that failed single-shot
    # Binomial test: is repair rate significantly > 0?
    if failed_initially > 0:
        repair_binom = stats.binomtest(
            repair_count, failed_initially, 0.0001, alternative="greater"
        )
        repair_p = repair_binom.pvalue
    else:
        repair_p = np.nan
    results["repair"] = {
        "repair_count": repair_count,
        "failed_initially": failed_initially,
        "repair_rate": repair_count / failed_initially if failed_initially > 0 else 0.0,
        "binom_p_value": float(repair_p),
    }

    # --- Iterations to success ---
    ref_passed = ref[ref["phase2"] == 1]
    results["iterations"] = {
        "mean": float(ref_passed["iterations"].mean()) if len(ref_passed) > 0 else np.nan,
        "median": float(ref_passed["iterations"].median()) if len(ref_passed) > 0 else np.nan,
        "std": float(ref_passed["iterations"].std()) if len(ref_passed) > 0 else np.nan,
    }

    return results


def run_multi_model_comparison(
    df: pd.DataFrame,
    config: AnalysisConfig,
    metric: str = "phase2",
) -> dict[str, Any]:
    """Friedman test comparing all models under the second condition.

    Uses config.models for iteration order, and the last condition
    from config (typically "refinement" or its variant).
    """
    # Use the treatment condition: last condition present in the data
    data_conditions = sorted(df["condition"].unique().tolist())
    treatment = data_conditions[-1] if data_conditions else config.conditions[-1]
    ref = df[df["condition"] == treatment]

    # Use models actually present in data, not config (names may differ)
    groups = []
    model_names = []
    for model in sorted(ref["model"].unique()):
        sub = ref[ref["model"] == model]
        if sub.empty:
            continue
        # Average across seeds per op
        pivoted = sub.pivot_table(index="op_name", values=metric, aggfunc="mean")
        groups.append(pivoted[metric].values)
        model_names.append(model)

    if len(groups) < 3:
        return {"error": f"need at least 3 models for Friedman test, got {len(groups)}"}

    # Align to common ops
    common_len = min(len(g) for g in groups)
    groups = [g[:common_len] for g in groups]

    return {
        "metric": metric,
        "models": model_names,
        "friedman": friedman_test(groups),
    }


def aggregate_across_seeds(
    df: pd.DataFrame,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Run analysis per seed, then aggregate (mean +/- SD) as per paper.

    Uses config to drive per-seed calls to run_pairwise_analysis.
    """
    all_results: dict[str, list[dict]] = {}

    seeds_present = sorted(df["seed"].unique())
    models_present = sorted(df["model"].unique())

    for model in models_present:
        per_seed_results = []
        for seed in seeds_present:
            result = run_pairwise_analysis(df, model, config, seed)
            if "error" not in result:
                per_seed_results.append(result)
        all_results[model] = per_seed_results

    # Aggregate
    summary: dict[str, Any] = {}
    for model, seed_results in all_results.items():
        if not seed_results:
            continue

        summary[model] = {}
        for phase in ["phase1", "phase2"]:
            rates_ss = [r[phase]["rate_single_shot"] for r in seed_results]
            rates_ref = [r[phase]["rate_refinement"] for r in seed_results]
            deltas = [r[phase]["delta"] for r in seed_results]
            p_values = [r[phase]["mcnemar"]["p_value"] for r in seed_results]
            h_values = [r[phase]["cohens_h"] for r in seed_results]

            summary[model][phase] = {
                "rate_single_shot": f"{np.mean(rates_ss):.3f} +/- {np.std(rates_ss):.3f}",
                "rate_refinement": f"{np.mean(rates_ref):.3f} +/- {np.std(rates_ref):.3f}",
                "delta": f"{np.mean(deltas):.3f} +/- {np.std(deltas):.3f}",
                "p_value_mean": float(np.mean(p_values)),
                "cohens_h_mean": float(np.mean(h_values)),
                "effect_interpretation": interpret_effect_size(float(np.mean(h_values))),
            }

        # Repair stats
        repair_rates = [r["repair"]["repair_rate"] for r in seed_results]
        summary[model]["repair"] = {
            "rate": f"{np.mean(repair_rates):.3f} +/- {np.std(repair_rates):.3f}",
        }

        # Iterations
        iter_means = [
            r["iterations"]["mean"]
            for r in seed_results
            if not np.isnan(r["iterations"]["mean"])
        ]
        if iter_means:
            summary[model]["iterations"] = {
                "mean": f"{np.mean(iter_means):.2f} +/- {np.std(iter_means):.2f}",
            }

    return summary


# ---------------------------------------------------------------------------
# Multiple comparisons correction across all tests
# ---------------------------------------------------------------------------


def apply_holm_correction(
    all_results: dict[str, dict],
    alpha: float,
) -> dict[str, dict]:
    """Collect all p-values and apply Holm-Bonferroni correction.

    Takes alpha directly instead of reading a global constant.
    """
    p_values = []
    labels = []

    for model, result in all_results.items():
        if "error" in result:
            continue
        for phase in ["phase1", "phase2"]:
            if phase in result:
                p_values.append(result[phase]["mcnemar"]["p_value"])
                labels.append(f"{model}_{phase}_mcnemar")
        if "runtime" in result and not np.isnan(result["runtime"]["wilcoxon"]["p_value"]):
            p_values.append(result["runtime"]["wilcoxon"]["p_value"])
            labels.append(f"{model}_runtime_wilcoxon")
        if "speedup" in result and not np.isnan(result["speedup"]["wilcoxon"]["p_value"]):
            p_values.append(result["speedup"]["wilcoxon"]["p_value"])
            labels.append(f"{model}_speedup_wilcoxon")

    if not p_values:
        return {}

    corrected = holm_bonferroni(p_values)
    return {
        label: {"raw_p": raw, "corrected_p": corr, "significant": corr < alpha}
        for label, raw, corr in zip(labels, p_values, corrected)
    }
