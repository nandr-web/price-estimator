"""Run estimator bias analysis and generate reports.

Trains a debiased model, computes per-estimator bias profiles, and
generates visualizations.

Usage:
    python scripts/bias_analysis.py --data resources/aora_historical_quotes.csv --output outputs/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="muted")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from price_estimator.bias import compute_estimator_bias, format_bias_report  # noqa: E402
from price_estimator.data import load_data, validate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def plot_bias_summary(bias_results: dict, output_dir: Path) -> None:
    """Bar chart of mean bias per estimator with confidence intervals."""
    summary = bias_results["summary"]

    fig, ax = plt.subplots(figsize=(8, 5))

    estimators = sorted(summary.keys())
    means = [summary[e]["mean_pct_bias"] for e in estimators]
    ci_lows = [summary[e]["ci_95_low"] for e in estimators]
    ci_highs = [summary[e]["ci_95_high"] for e in estimators]
    labels = [summary[e]["label"].upper() for e in estimators]

    colors = []
    for label in labels:
        if label == "SAFE":
            colors.append("steelblue")
        elif label == "AGGRESSIVE":
            colors.append("coral")
        else:
            colors.append("gray")

    errors_low = [m - lo for m, lo in zip(means, ci_lows)]
    errors_high = [h - m for m, h in zip(means, ci_highs)]

    bars = ax.bar(
        estimators,
        means,
        yerr=[errors_low, errors_high],
        capsize=6,
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Mean Bias (%)")
    ax.set_title("Estimator Bias: Mean % Deviation from Neutral Price")

    for bar, mean, label in zip(bars, means, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5 if mean >= 0 else bar.get_height() - 1.5,
            f"{mean:+.1f}% [{label}]",
            ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_dir / "bias_summary.png", dpi=150)
    plt.close(fig)


def plot_bias_by_part_type(bias_results: dict, output_dir: Path) -> None:
    """Heatmap of bias by estimator x part type."""
    by_pt = bias_results["by_part_type"]

    # Build matrix
    estimators = sorted(by_pt.keys())
    all_pts = set()
    for est_data in by_pt.values():
        all_pts.update(est_data.keys())
    part_types = sorted(all_pts)

    matrix = np.zeros((len(estimators), len(part_types)))
    for i, est in enumerate(estimators):
        for j, pt in enumerate(part_types):
            if pt in by_pt[est]:
                matrix[i, j] = by_pt[est][pt]["mean_pct_bias"]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pd.DataFrame(matrix, index=estimators, columns=part_types),
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
    )
    ax.set_title("Estimator Bias by Part Type (% deviation from neutral)")
    fig.tight_layout()
    fig.savefig(output_dir / "bias_by_part_type.png", dpi=150)
    plt.close(fig)


def plot_bias_over_time(bias_results: dict, output_dir: Path) -> None:
    """Line plot of monthly bias per estimator."""
    over_time = bias_results["over_time"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for est in sorted(over_time.keys()):
        months = sorted(over_time[est].keys())
        biases = [over_time[est][m]["mean_pct_bias"] for m in months]
        ax.plot(months, biases, marker="o", label=est, linewidth=1.5, markersize=4)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Bias (%)")
    ax.set_title("Estimator Bias Over Time")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "bias_over_time.png", dpi=150)
    plt.close(fig)


def main(data_path: str, output_path: str) -> None:
    """Run the full bias analysis pipeline."""
    data_path = Path(data_path)
    output_path = Path(output_path)
    results_dir = output_path / "results"
    figures_dir = output_path / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_data(data_path)
    validate(df)
    print(f"  {len(df)} rows loaded")

    # Run bias analysis
    print("\nComputing estimator bias...")
    bias_results = compute_estimator_bias(df)

    # Print report
    report = format_bias_report(bias_results)
    print(f"\n{report}")

    # Save results
    json_path = results_dir / "estimator_bias.json"
    with open(json_path, "w") as f:
        json.dump(bias_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Generate plots
    print("Generating plots...")
    plot_bias_summary(bias_results, figures_dir)
    plot_bias_by_part_type(bias_results, figures_dir)
    plot_bias_over_time(bias_results, figures_dir)
    print(f"Figures saved to {figures_dir}/")

    print("\nBias analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run estimator bias analysis")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--output", required=True, help="Base output directory")
    args = parser.parse_args()
    main(args.data, args.output)
