"""LaTeX table generators and matplotlib plots for experiment results."""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------


def results_to_latex_main_table(all_results: dict[str, dict], alpha: float) -> str:
    """Generate the main results table in LaTeX."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Phase 1 and Phase 2 pass rates by model and condition}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{l cc cc cc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Phase 1}} & \multicolumn{2}{c}{\textbf{Phase 2}} & \multicolumn{2}{c}{\textbf{Effect}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"\textbf{Model} & Single & Refine & Single & Refine & $h_1$ & $h_2$ \\",
        r"\midrule",
    ]

    for model, result in all_results.items():
        if "error" in result:
            continue
        p1 = result["phase1"]
        p2 = result["phase2"]
        sig1 = "*" if p1["mcnemar"]["p_value"] < alpha else ""
        sig2 = "*" if p2["mcnemar"]["p_value"] < alpha else ""
        lines.append(
            f"{model} & {p1['rate_single_shot']:.3f} & {p1['rate_refinement']:.3f}{sig1} "
            f"& {p2['rate_single_shot']:.3f} & {p2['rate_refinement']:.3f}{sig2} "
            f"& {p1['cohens_h']:.3f} & {p2['cohens_h']:.3f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def results_to_latex_repair_table(all_results: dict[str, dict]) -> str:
    """Generate repair success rate table."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Repair success and iteration statistics}",
        r"\label{tab:repair}",
        r"\begin{tabular}{l ccc cc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Failed} & \textbf{Repaired} & \textbf{Repair \%} & \textbf{Iter mean} & \textbf{Iter median} \\",
        r"\midrule",
    ]

    for model, result in all_results.items():
        if "error" in result:
            continue
        rep = result["repair"]
        it = result["iterations"]
        lines.append(
            f"{model} & {rep['failed_initially']} & {rep['repair_count']} "
            f"& {rep['repair_rate']:.3f} & {it['mean']:.2f} & {it['median']:.1f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def results_to_latex_correction_table(corrections: dict[str, dict]) -> str:
    """Generate Holm-Bonferroni correction table."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Holm-Bonferroni corrected p-values}",
        r"\label{tab:holm}",
        r"\begin{tabular}{l cc c}",
        r"\toprule",
        r"\textbf{Test} & \textbf{Raw $p$} & \textbf{Corrected $p$} & \textbf{Sig.} \\",
        r"\midrule",
    ]

    for label, vals in sorted(corrections.items()):
        sig = r"$\checkmark$" if vals["significant"] else ""
        lines.append(
            f"{label.replace('_', r'\\_')} & {vals['raw_p']:.4f} & {vals['corrected_p']:.4f} & {sig} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_pass_rates(all_results: dict[str, dict], output_dir: Path, alpha: float) -> None:
    """Bar chart: Phase 1/2 pass rates by model and condition."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)

    models_present = [m for m in all_results if "error" not in all_results[m]]
    if not models_present:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, phase in enumerate(["phase1", "phase2"]):
        ax = axes[idx]
        x = np.arange(len(models_present))
        width = 0.35

        ss_rates = [all_results[m][phase]["rate_single_shot"] for m in models_present]
        ref_rates = [all_results[m][phase]["rate_refinement"] for m in models_present]

        ax.bar(x - width / 2, ss_rates, width, label="Single-shot", color="#4C72B0")
        ax.bar(x + width / 2, ref_rates, width, label="Refinement", color="#DD8452")

        ax.set_xlabel("Model")
        ax.set_ylabel("Pass Rate")
        ax.set_title(f"Phase {idx + 1} Pass Rate")
        ax.set_xticks(x)
        ax.set_xticklabels([m.split("-")[0] for m in models_present], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.legend()

        # Significance markers
        for i, m in enumerate(models_present):
            p = all_results[m][phase]["mcnemar"]["p_value"]
            if p < alpha:
                max_h = max(ss_rates[i], ref_rates[i])
                ax.text(i, max_h + 0.02, "*", ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "pass_rates.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "pass_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pass_rates.pdf/png to {output_dir}")


def plot_iteration_convergence(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot: cumulative pass rate by refinement iteration."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)

    ref = df[df["condition"] == "refinement"]
    models_present = sorted(ref["model"].unique().tolist())
    if not models_present:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    max_iters = int(ref["iterations"].max()) if not ref.empty else 5  # type: ignore[union-attr]

    for model in models_present:
        sub = ref[ref["model"] == model]
        cumulative = []
        for i in range(1, max_iters + 1):
            passed = sub[sub["iterations"] <= i]["phase2"].sum()
            total = len(sub)
            cumulative.append(passed / total if total > 0 else 0)
        ax.plot(range(1, max_iters + 1), cumulative, marker="o", label=model.split("-")[0])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Phase 2 Pass Rate")
    ax.set_title("Convergence by Refinement Iteration")
    ax.legend()
    ax.set_xticks(range(1, max_iters + 1))
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_dir / "convergence.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved convergence.pdf/png to {output_dir}")
