"""Estimator bias analysis for the price estimator.

Trains a debiased model (no estimator feature), computes residuals
per estimator, and identifies who is the "safe" (high) vs "aggressive"
(low) quoter. Also checks whether bias is uniform or varies by job type.
"""

import logging

import numpy as np
import pandas as pd

from price_estimator.features import extract_description_features
from price_estimator.models import XGBoostModel

logger = logging.getLogger(__name__)


def compute_estimator_bias(df: pd.DataFrame) -> dict:
    """Compute estimator bias using a debiased model (M9).

    Trains an XGBoost model without the estimator feature to get a
    "neutral" price for each quote. The difference between actual price
    and neutral prediction is the estimator's bias.

    Args:
        df: Full cleaned dataset.

    Returns:
        Dictionary with:
            - summary: per-estimator bias stats (mean, median, CI, label)
            - by_part_type: bias breakdown by estimator x part type
            - by_material: bias breakdown by estimator x material
            - over_time: monthly rolling bias per estimator
    """
    # Train debiased model on full dataset
    m9 = XGBoostModel(name="M9_bias", exclude_estimator=True)
    m9.fit(df)
    neutral_preds = m9.predict(df)

    # Clamp predictions to a minimum to avoid division by zero
    neutral_preds = np.maximum(neutral_preds, 1.0)

    # Compute residuals (actual - predicted) as % of actual price
    residuals = df["TotalPrice_USD"].values - neutral_preds
    pct_residuals = residuals / df["TotalPrice_USD"].values * 100

    df_analysis = df.assign(
        residual=residuals,
        pct_residual=pct_residuals,
        neutral_pred=neutral_preds,
    )

    # --- Per-estimator summary ---
    summary = {}
    for est in sorted(df["Estimator"].unique()):
        mask = df_analysis["Estimator"] == est
        est_residuals = df_analysis.loc[mask, "pct_residual"].values

        # Bootstrap 95% CI for mean bias
        ci_low, ci_high = _bootstrap_ci(est_residuals)

        mean_bias = float(np.mean(est_residuals))
        median_bias = float(np.median(est_residuals))

        # Label: safe (positive = quotes high), aggressive (negative = quotes low)
        if ci_low > 0:
            label = "safe"
        elif ci_high < 0:
            label = "aggressive"
        else:
            label = "neutral"

        summary[est] = {
            "mean_pct_bias": mean_bias,
            "median_pct_bias": median_bias,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "std_pct_bias": float(np.std(est_residuals)),
            "n_quotes": int(mask.sum()),
            "label": label,
        }

    # --- Bias by part type ---
    desc_features = extract_description_features(df)
    df_analysis = df_analysis.assign(base_part_type=desc_features["base_part_type"])

    by_part_type = {}
    for est in sorted(df["Estimator"].unique()):
        est_data = df_analysis[df_analysis["Estimator"] == est]
        pt_biases = {}
        for pt in est_data["base_part_type"].unique():
            pt_mask = est_data["base_part_type"] == pt
            vals = est_data.loc[pt_mask, "pct_residual"].values
            pt_biases[pt] = {
                "mean_pct_bias": float(np.mean(vals)),
                "n_quotes": int(len(vals)),
            }
        by_part_type[est] = pt_biases

    # --- Bias by material ---
    by_material = {}
    for est in sorted(df["Estimator"].unique()):
        est_data = df_analysis[df_analysis["Estimator"] == est]
        mat_biases = {}
        for mat in est_data["Material"].dropna().unique():
            mat_mask = est_data["Material"] == mat
            vals = est_data.loc[mat_mask, "pct_residual"].values
            mat_biases[mat] = {
                "mean_pct_bias": float(np.mean(vals)),
                "n_quotes": int(len(vals)),
            }
        by_material[est] = mat_biases

    # --- Bias over time ---
    df_analysis = df_analysis.assign(year_month=df_analysis["Date"].dt.to_period("M").astype(str))
    over_time = {}
    for est in sorted(df["Estimator"].unique()):
        est_data = df_analysis[df_analysis["Estimator"] == est]
        monthly = est_data.groupby("year_month")["pct_residual"].agg(["mean", "count"])
        over_time[est] = {
            row.Index: {"mean_pct_bias": float(row.mean), "n_quotes": int(row.count)}
            for row in monthly.itertuples()
        }

    return {
        "summary": summary,
        "by_part_type": by_part_type,
        "by_material": by_material,
        "over_time": over_time,
    }


def _bootstrap_ci(
    values: np.ndarray, n_bootstrap: int = 5000, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        values: Array of observations.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound) for the mean.
    """
    rng = np.random.default_rng(42)
    boot_means = np.array(
        [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    )
    alpha = 1 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def format_bias_report(bias_results: dict) -> str:
    """Format bias results as a human-readable report.

    Args:
        bias_results: Output from compute_estimator_bias().

    Returns:
        Formatted string report.
    """
    lines = ["=" * 60, "ESTIMATOR BIAS ANALYSIS", "=" * 60, ""]

    for est, est_stats in sorted(
        bias_results["summary"].items(),
        key=lambda x: x[1]["mean_pct_bias"],
        reverse=True,
    ):
        label = est_stats["label"].upper()
        lines.append(f"{est} [{label}]")
        lines.append(f"  Mean bias: {est_stats['mean_pct_bias']:+.1f}%")
        lines.append(f"  Median bias: {est_stats['median_pct_bias']:+.1f}%")
        lines.append(
            f"  95% CI: [{est_stats['ci_95_low']:+.1f}%, {est_stats['ci_95_high']:+.1f}%]"
        )
        lines.append(f"  Quotes: {est_stats['n_quotes']}")
        lines.append("")

    # Top part-type biases per estimator
    lines.append("-" * 60)
    lines.append("BIAS BY PART TYPE (top deviations)")
    lines.append("-" * 60)
    for est, pt_biases in sorted(bias_results["by_part_type"].items()):
        lines.append(f"\n{est}:")
        sorted_pts = sorted(
            pt_biases.items(), key=lambda x: abs(x[1]["mean_pct_bias"]), reverse=True
        )
        for pt, pt_stats in sorted_pts[:5]:
            lines.append(f"  {pt}: {pt_stats['mean_pct_bias']:+.1f}% (n={pt_stats['n_quotes']})")

    return "\n".join(lines)
