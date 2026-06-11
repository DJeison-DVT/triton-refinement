"""Statistical primitives for Triton Refinement experiment analysis.

All functions extracted from scripts/analyze.py. Every function takes its
parameters explicitly -- no global constants like ALPHA or N_OPS.

Implements the statistical plan from paper/reporteEstadistico.tex:
- McNemar test for paired binary metrics
- Wilcoxon signed-rank test for paired continuous metrics
- Friedman test for multi-group paired comparison
- Cohen's d / Cohen's h effect sizes
- Wilson score and bootstrap confidence intervals
- A priori power analysis for McNemar and Wilcoxon
- Holm-Bonferroni multiple-comparison correction
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Statistical tests (matching reporteEstadistico.tex Section 9)
# ---------------------------------------------------------------------------


def mcnemar_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """McNemar test for paired binary data.

    a, b: arrays of 0/1 per operation (same length, paired by op).
    One-sided: tests whether *b* (refinement) is better than *a* (single-shot).

    Returns dict with statistic, p_value, and contingency counts.
    """
    # Contingency table for discordant pairs
    # b=1  b=0
    # a=1 [ n11  n10 ]
    # a=0 [ n01  n00 ]
    _n11 = int(np.sum((a == 1) & (b == 1)))  # noqa: F841 — concordant, kept for documentation
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    _n00 = int(np.sum((a == 0) & (b == 0)))  # noqa: F841 — concordant, kept for documentation

    # Discordant pairs
    n_discordant = n10 + n01
    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n10": n10, "n01": n01}

    # Use exact binomial test when discordant pairs < 25
    if n_discordant < 25:
        # One-sided: refinement (b) better than single-shot (a) means n01 > n10
        p_value = stats.binomtest(n01, n_discordant, 0.5, alternative="greater").pvalue
    else:
        # Chi-squared approximation with continuity correction
        chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        # One-sided: divide by 2 if in expected direction
        if n01 > n10:
            p_value = p_value / 2
        else:
            p_value = 1 - p_value / 2

    return {
        "statistic": float((n10 - n01) ** 2 / (n10 + n01)) if n_discordant > 0 else 0.0,
        "p_value": float(p_value),
        "n10": n10,
        "n01": n01,
    }


def wilcoxon_test(
    a: np.ndarray, b: np.ndarray, alternative: str = "two-sided"
) -> dict[str, float]:
    """Wilcoxon signed-rank test for paired continuous data.

    alternative: "two-sided", "less" (b < a), or "greater" (b > a).
    Returns dict with statistic, p_value, n_valid.
    """
    # Remove pairs where both are NaN or equal
    mask = ~(np.isnan(a) | np.isnan(b))
    a_clean, b_clean = a[mask], b[mask]
    diff = b_clean - a_clean
    diff = diff[diff != 0]

    if len(diff) < 10:
        return {"statistic": np.nan, "p_value": np.nan, "n_valid": len(diff)}

    stat, p_value = stats.wilcoxon(diff, alternative=alternative)
    return {"statistic": float(stat), "p_value": float(p_value), "n_valid": len(diff)}


def friedman_test(groups: list[np.ndarray]) -> dict[str, float]:
    """Friedman test for comparing multiple paired groups.

    groups: list of arrays, one per condition, all same length (paired by op).
    """
    # Remove ops where any group has NaN
    stacked = np.column_stack(groups)
    mask = ~np.any(np.isnan(stacked), axis=1)
    stacked = stacked[mask]

    if stacked.shape[0] < 10:
        return {"statistic": np.nan, "p_value": np.nan, "n_valid": stacked.shape[0]}

    stat, p_value = stats.friedmanchisquare(
        *[stacked[:, i] for i in range(stacked.shape[1])]
    )
    return {"statistic": float(stat), "p_value": float(p_value), "n_valid": stacked.shape[0]}


# ---------------------------------------------------------------------------
# Effect sizes (matching reporteEstadistico.tex Sections 6-6.1)
# ---------------------------------------------------------------------------


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired continuous data."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a_clean, b_clean = a[mask], b[mask]
    if len(a_clean) < 2:
        return np.nan
    diff = b_clean - a_clean
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for comparing two proportions.

    h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))
    """
    return float(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d/h magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Parameters
    ----------
    successes : int
        Number of successes.
    n : int
        Total number of trials.
    alpha : float
        Significance level (default 0.05 for a 95 % CI).
    """
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, float(center - margin)), min(1.0, float(center + margin)))


def bootstrap_ci_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap CI for the difference in means (b - a).

    Parameters
    ----------
    a, b : np.ndarray
        Paired arrays (may contain NaN; NaN-pairs are dropped).
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (default 0.05 for a 95 % CI).
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    a_clean, b_clean = a[mask], b[mask]
    n = len(a_clean)
    if n < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(42)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(b_clean[idx]) - np.mean(a_clean[idx])

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# A priori power analysis (reporteEstadistico.tex Section 7)
# ---------------------------------------------------------------------------


def power_analysis_mcnemar(
    n: int = 166, alpha: float = 0.05, power: float = 0.80
) -> dict[str, Any]:
    """A priori power analysis for McNemar test (paired proportions).

    Estimates the minimum detectable difference in discordant proportions
    for *n* paired observations at given *alpha* and *power*.

    Parameters
    ----------
    n : int
        Number of paired observations (default 166, the TritonBench-T count).
    alpha : float
        Significance level.
    power : float
        Target statistical power.
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    # For McNemar, power depends on discordant pairs.
    # Minimum discordant proportion detectable:
    # n_discordant >= (z_alpha + z_beta)^2 / (effect^2)
    # With medium effect (h=0.5), the required n for a sign test:
    min_n_sign = ((z_alpha + z_beta) / 0.5) ** 2
    return {
        "n_available": n,
        "alpha": alpha,
        "target_power": power,
        "min_n_for_medium_effect": math.ceil(min_n_sign),
        "sufficient": n >= min_n_sign,
    }


def power_analysis_wilcoxon(
    n: int = 166, alpha: float = 0.05, power: float = 0.80
) -> dict[str, Any]:
    """A priori power analysis for Wilcoxon signed-rank test.

    Uses normal approximation: n >= ((z_alpha + z_beta) / d)^2
    for paired design with Cohen's d as effect size.

    Parameters
    ----------
    n : int
        Number of paired observations (default 166).
    alpha : float
        Significance level.
    power : float
        Target statistical power.
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    results: dict[str, Any] = {}
    for label, d in [("small", 0.2), ("medium", 0.5), ("large", 0.8)]:
        min_n = math.ceil(((z_alpha + z_beta) / d) ** 2)
        results[label] = {
            "effect_size": d,
            "min_n_required": min_n,
            "sufficient": n >= min_n,
        }

    results["n_available"] = n
    results["alpha"] = alpha
    results["target_power"] = power
    return results


# ---------------------------------------------------------------------------
# Multiple comparison correction (reporteEstadistico.tex Section 9.4)
# ---------------------------------------------------------------------------


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values.

    Returns corrected p-values (same order as input).
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        adjusted = min(adjusted, 1.0)
        cummax = max(cummax, adjusted)
        corrected[orig_idx] = cummax
    return corrected
