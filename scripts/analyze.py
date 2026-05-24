"""CLI entry point for statistical analysis of Triton Refinement experiments.

Thin wrapper that loads config, builds the DataFrame, runs the analysis
pipeline, and prints results. All logic lives in the ``analysis`` package.

Usage:
    python scripts/analyze.py --results-dir results/trajectories
    python scripts/analyze.py --results-dir results/trajectories --latex --plot
    python scripts/analyze.py --config experiment_config.json --latex --json results/analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from analysis.config import load_config
from analysis.data import build_experiment_df
from analysis.output import (
    plot_iteration_convergence,
    plot_pass_rates,
    results_to_latex_correction_table,
    results_to_latex_main_table,
    results_to_latex_repair_table,
)
from analysis.pipeline import (
    aggregate_across_seeds,
    apply_holm_correction,
    run_multi_model_comparison,
    run_pairwise_analysis,
)
from analysis.stats import (
    interpret_effect_size,
    power_analysis_mcnemar,
    power_analysis_wilcoxon,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical analysis for Triton Refinement")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing experiment run folders")
    parser.add_argument("--config", type=Path, default=Path("experiment_config.json"),
                        help="Path to experiment_config.json")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"),
                        help="Directory for figures and tables")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX tables to stdout")
    parser.add_argument("--json", type=Path, default=None, help="Write full results to JSON file")
    parser.add_argument("--plot", action="store_true", help="Generate plots (requires matplotlib)")
    parser.add_argument("--local", type=Path, default=None,
                        help="Trajectory-only analysis on a flat directory of JSONL files "
                             "(e.g. results/local/trajectories). Skips full experiment pipeline.")
    args = parser.parse_args()

    # --- Local trajectory-only mode (week 2) ---
    if args.local is not None:
        from analysis.trajectory import analyze_trajectories, print_trajectory_report
        results = analyze_trajectories(args.local)
        print_trajectory_report(results)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults written to {args.json}")
        return

    # --- Full experiment mode (week 3+) ---
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.results_dir}...")
    df = build_experiment_df(args.results_dir, config.conditions)

    if df.empty:
        print("No data found. Place experiment results in the expected directory structure.")
        print("Expected: {results_dir}/{model}_{condition}_seed{seed}/trajectories/{op}.jsonl")
        sys.exit(1)

    print(f"Loaded {len(df)} records: {df['model'].nunique()} models, "
          f"{df['seed'].nunique()} seeds, {df['op_name'].nunique()} ops")

    # --- Skip rate diagnostic ---
    if "phases_skipped" in df.columns:
        total_rows = len(df)
        test_skipped = sum(1 for _, row in df.iterrows() if "test" in row["phases_skipped"])
        review_skipped = sum(1 for _, row in df.iterrows() if "review" in row["phases_skipped"])
        print(f"  Phase skip rates: test={test_skipped}/{total_rows} ({test_skipped/total_rows:.1%}), "
              f"review={review_skipped}/{total_rows} ({review_skipped/total_rows:.1%})")

    # --- A priori power analysis ---
    if "power_analysis" in config.steps:
        print("\n=== A Priori Power Analysis ===")
        pa_mcnemar = power_analysis_mcnemar(n=config.n_ops, alpha=config.alpha)
        print(f"  McNemar: n={pa_mcnemar['n_available']}, "
              f"min_n(medium)={pa_mcnemar['min_n_for_medium_effect']}, "
              f"sufficient={pa_mcnemar['sufficient']}")
        pa_wilcoxon = power_analysis_wilcoxon(n=config.n_ops, alpha=config.alpha)
        for label in ["small", "medium", "large"]:
            w = pa_wilcoxon[label]
            print(f"  Wilcoxon ({label} d={w['effect_size']}): "
                  f"min_n={w['min_n_required']}, sufficient={w['sufficient']}")

    # --- Per-model pairwise analysis ---
    all_results: dict[str, dict] = {}
    if "pairwise" in config.steps:
        print("\n=== Pairwise Analysis (single-shot vs refinement) ===")
        for model in sorted(df["model"].unique()):
            result = run_pairwise_analysis(df, model, config)
            all_results[model] = result

            if "error" in result:
                print(f"\n{model}: {result['error']}")
                continue

            print(f"\n--- {model} (n={result['n_ops']}) ---")
            for phase in ["phase1", "phase2"]:
                p = result[phase]
                sig = "***" if p["mcnemar"]["p_value"] < 0.001 else (
                    "**" if p["mcnemar"]["p_value"] < 0.01 else (
                    "*" if p["mcnemar"]["p_value"] < config.alpha else ""))
                print(f"  {phase}: {p['rate_single_shot']:.3f} -> {p['rate_refinement']:.3f} "
                      f"(delta={p['delta']:+.3f}, h={p['cohens_h']:.3f} "
                      f"[{interpret_effect_size(p['cohens_h'])}], "
                      f"p={p['mcnemar']['p_value']:.4f}{sig})")

            rep = result["repair"]
            print(f"  repair: {rep['repair_count']}/{rep['failed_initially']} "
                  f"({rep['repair_rate']:.3f})")

            it = result["iterations"]
            print(f"  iterations to success: mean={it['mean']:.2f}, median={it['median']:.1f}")

    # --- Multi-model comparison ---
    if "friedman" in config.steps:
        print("\n=== Friedman Test (multi-model comparison) ===")
        for metric in ["phase2", "phase1"]:
            fm = run_multi_model_comparison(df, config, metric)
            if "error" not in fm:
                print(f"  {metric}: chi2={fm['friedman']['statistic']:.3f}, "
                      f"p={fm['friedman']['p_value']:.4f}")
            else:
                print(f"  {metric}: {fm['error']}")

    # --- Holm-Bonferroni correction ---
    corrections: dict[str, dict] = {}
    if "holm_correction" in config.steps and all_results:
        print("\n=== Holm-Bonferroni Correction ===")
        corrections = apply_holm_correction(all_results, config.alpha)
        for label, vals in sorted(corrections.items()):
            sig = "SIG" if vals["significant"] else "ns"
            print(f"  {label}: raw={vals['raw_p']:.4f} -> "
                  f"corrected={vals['corrected_p']:.4f} [{sig}]")

    # --- Cross-seed aggregation ---
    if "cross_seed_aggregation" in config.steps and df["seed"].nunique() > 1:
        print("\n=== Cross-Seed Aggregation (mean +/- SD) ===")
        summary = aggregate_across_seeds(df, config)
        for model, data in summary.items():
            print(f"\n--- {model} ---")
            for key, vals in data.items():
                print(f"  {key}: {vals}")

    # --- LaTeX output ---
    if args.latex and all_results:
        print("\n=== LaTeX Tables ===\n")
        print(results_to_latex_main_table(all_results, config.alpha))
        print()
        print(results_to_latex_repair_table(all_results))
        print()
        if corrections:
            print(results_to_latex_correction_table(corrections))

    # --- Plots ---
    if args.plot and all_results:
        print("\n=== Generating Plots ===")
        plot_pass_rates(all_results, args.output_dir, config.alpha)
        plot_iteration_convergence(df, args.output_dir)

    # --- JSON dump ---
    if args.json:
        def convert(obj: object) -> object:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "pairwise": all_results,
                "corrections": corrections,
            }, f, indent=2, default=convert)
        print(f"\nFull results written to {args.json}")


if __name__ == "__main__":
    main()
