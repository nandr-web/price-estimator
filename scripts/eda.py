"""EDA script for the price estimator.

Thin CLI wrapper that calls analysis functions from src/price_estimator/
and generates plots. All numeric results are written to
outputs/results/data_profile.json. All figures are written to outputs/figures/.

Usage:
    python scripts/eda.py --data resources/aora_historical_quotes.csv --output outputs/
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

from price_estimator.analysis import (  # noqa: E402
    compute_confounding_analysis,
    compute_lead_time_analysis,
    compute_rush_premium,
    compute_summary_stats,
    compute_unit_price_analysis,
    compute_volume_discount,
)
from price_estimator.data import (  # noqa: E402
    compute_unit_price,
    get_missing_report,
    load_data,
    validate,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (10, 6)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_price_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Histogram of TotalPrice_USD."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(df["TotalPrice_USD"], bins=40, edgecolor="black", alpha=0.7)
    ax.set_xlabel("TotalPrice_USD ($)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Total Price")
    ax.axvline(df["TotalPrice_USD"].mean(), color="red", linestyle="--", label="Mean")
    ax.axvline(df["TotalPrice_USD"].median(), color="orange", linestyle="--", label="Median")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "eda_price_distribution.png", dpi=150)
    plt.close(fig)


def plot_unit_price_by_category(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar charts of unit price by material, part type, process, estimator."""
    unit_price = compute_unit_price(df)
    plot_df = df.assign(unit_price=unit_price)

    categories = [
        ("Material", "eda_unit_price_by_material.png"),
        ("Process", "eda_unit_price_by_process.png"),
        ("Estimator", "eda_unit_price_by_estimator.png"),
    ]

    for col, filename in categories:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        order = plot_df.dropna(subset=[col]).groupby(col)["unit_price"].mean().sort_values().index
        sns.barplot(data=plot_df.dropna(subset=[col]), x=col, y="unit_price", order=order, ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel("Mean Unit Price ($)")
        ax.set_title(f"Unit Price by {col}")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    # Part type — horizontal bar for readability
    fig, ax = plt.subplots(figsize=(10, 8))
    base_types = plot_df["PartDescription"].str.split(" - ").str[0]
    plot_df = plot_df.assign(base_part_type=base_types)
    order = plot_df.groupby("base_part_type")["unit_price"].mean().sort_values().index
    sns.barplot(data=plot_df, y="base_part_type", x="unit_price", order=order, ax=ax, orient="h")
    ax.set_ylabel("Part Type")
    ax.set_xlabel("Mean Unit Price ($)")
    ax.set_title("Unit Price by Part Type")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_unit_price_by_part_type.png", dpi=150)
    plt.close(fig)


def plot_volume_discount(df: pd.DataFrame, volume_results: dict, output_dir: Path) -> None:
    """Scatter of log(qty) vs log(unit_price) with global + per-part-type fitted lines."""
    unit_price = compute_unit_price(df)
    log_qty = np.log(df["Quantity"].astype(float))
    log_up = np.log(unit_price)
    base_types = df["PartDescription"].str.split(" - ").str[0]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Scatter colored by part type
    for pt in sorted(base_types.unique()):
        mask = base_types == pt
        ax.scatter(log_qty[mask], log_up[mask], alpha=0.4, s=20, label=pt)

    # Global fitted line
    x_line = np.linspace(log_qty.min(), log_qty.max(), 100)
    slope = volume_results["slope"]
    intercept = volume_results["intercept"]
    r_sq = volume_results["r_squared"]
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label=f"Global: slope={slope:.3f}, R\u00b2={r_sq:.3f}",
    )

    # Per-part-type fitted lines
    per_part = volume_results.get("per_part_type", {})
    for pt, pt_stats in per_part.items():
        y_line = pt_stats["slope"] * x_line + pt_stats["intercept"]
        ax.plot(x_line, y_line, linewidth=1, alpha=0.5)

    ax.set_xlabel("log(Quantity)")
    ax.set_ylabel("log(Unit Price)")
    ax.set_title("Volume Discount: log(Qty) vs log(Unit Price)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_volume_discount.png", dpi=150)
    plt.close(fig)


def plot_rush_premium(df: pd.DataFrame, rush_results: dict, output_dir: Path) -> None:
    """Box plot of rush vs non-rush, plus per-category breakdown bar chart."""
    unit_price = compute_unit_price(df)
    plot_df = df.assign(unit_price=unit_price)
    plot_df["RushJob_label"] = plot_df["RushJob"].map({True: "Rush", False: "Standard"})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Overall box plot
    sns.boxplot(data=plot_df, x="RushJob_label", y="unit_price", ax=axes[0])
    axes[0].set_xlabel("Job Type")
    axes[0].set_ylabel("Unit Price ($)")
    axes[0].set_title("Rush Premium: Overall")

    # By part type
    by_pt = rush_results.get("by_part_type", {})
    if by_pt:
        pts = sorted(by_pt.keys())
        ratios = [by_pt[pt]["ratio"] for pt in pts]
        axes[1].barh(pts, ratios, color="steelblue", edgecolor="black")
        axes[1].axvline(1.0, color="gray", linestyle="--")
        axes[1].set_xlabel("Rush / Standard Ratio")
        axes[1].set_title("Rush Premium by Part Type")

    # By material
    by_mat = rush_results.get("by_material", {})
    if by_mat:
        mats = sorted(by_mat.keys())
        ratios = [by_mat[m]["ratio"] for m in mats]
        axes[2].barh(mats, ratios, color="coral", edgecolor="black")
        axes[2].axvline(1.0, color="gray", linestyle="--")
        axes[2].set_xlabel("Rush / Standard Ratio")
        axes[2].set_title("Rush Premium by Material")

    fig.tight_layout()
    fig.savefig(output_dir / "eda_rush_premium.png", dpi=150)
    plt.close(fig)


def plot_lead_time(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot of lead time vs unit price."""
    unit_price = compute_unit_price(df)
    corr, _ = stats.pearsonr(df["LeadTimeWeeks"], unit_price)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(df["LeadTimeWeeks"], unit_price, alpha=0.3, s=20)
    ax.set_xlabel("Lead Time (weeks)")
    ax.set_ylabel("Unit Price ($)")
    ax.set_title(f"Lead Time vs Unit Price (r={corr:.3f})")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_lead_time_vs_unit_price.png", dpi=150)
    plt.close(fig)


def plot_confounding_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of Estimator x PartType cross-tabs (counts and mean unit price)."""
    unit_price = compute_unit_price(df)
    analysis_df = df.assign(unit_price=unit_price)
    base_types = analysis_df["PartDescription"].str.split(" - ").str[0]
    analysis_df = analysis_df.assign(base_part_type=base_types)

    ct_counts = pd.crosstab(analysis_df["Estimator"], analysis_df["base_part_type"])
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(ct_counts, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Estimator x Part Type: Job Counts")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_confounding_counts.png", dpi=150)
    plt.close(fig)

    ct_price = analysis_df.pivot_table(
        values="unit_price", index="Estimator", columns="base_part_type", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(ct_price, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_title("Estimator x Part Type: Mean Unit Price ($)")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_confounding_unit_price.png", dpi=150)
    plt.close(fig)


def plot_missing_values(df: pd.DataFrame, missing_report: dict, output_dir: Path) -> None:
    """Visualize missing value patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    by_est = missing_report.get("by_estimator", {})
    if by_est:
        ax = axes[0]
        estimators = list(by_est.keys())
        counts = [by_est[e] for e in estimators]
        ax.bar(estimators, counts, color="coral", edgecolor="black")
        ax.set_xlabel("Estimator")
        ax.set_ylabel("Rows with Missing Values")
        ax.set_title("Missing Values by Estimator")
    else:
        axes[0].text(0.5, 0.5, "No missing values", ha="center", va="center")
        axes[0].set_title("Missing Values by Estimator")

    by_pt = missing_report.get("by_part_type", {})
    if by_pt:
        ax = axes[1]
        part_types = list(by_pt.keys())
        counts = [by_pt[p] for p in part_types]
        short_labels = [p.split(" - ")[0] for p in part_types]
        ax.barh(short_labels, counts, color="steelblue", edgecolor="black")
        ax.set_xlabel("Rows with Missing Values")
        ax.set_title("Missing Values by Part Type")
    else:
        axes[1].text(0.5, 0.5, "No missing values", ha="center", va="center")
        axes[1].set_title("Missing Values by Part Type")

    fig.tight_layout()
    fig.savefig(output_dir / "eda_missing_values.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(data_path: str, output_path: str) -> None:
    """Run the full EDA pipeline."""
    data_path = Path(data_path)
    output_path = Path(output_path)
    figures_dir = output_path / "figures"
    results_dir = output_path / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and validate ---
    print("Loading data...")
    df = load_data(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    print("Validating...")
    warnings = validate(df)
    for w in warnings:
        print(f"  WARNING: {w}")

    print("Analyzing missing values...")
    missing_report = get_missing_report(df)
    print(f"  Missing Material: {len(missing_report['missing_material_ids'])}")
    print(f"  Missing Process: {len(missing_report['missing_process_ids'])}")
    print(f"  Overlap (both missing): {missing_report['overlap_count']}")

    # --- Compute analyses ---
    print("Computing summary statistics...")
    summary = compute_summary_stats(df)

    print("Computing unit price analysis...")
    unit_price_analysis = compute_unit_price_analysis(df)

    print("Computing volume discount analysis...")
    volume_discount = compute_volume_discount(df)
    print(f"  Global slope: {volume_discount['slope']:.4f}")
    print(f"  Global R\u00b2: {volume_discount['r_squared']:.4f}")
    per_part = volume_discount.get("per_part_type", {})
    for pt, pt_stats in sorted(per_part.items()):
        print(f"  {pt}: slope={pt_stats['slope']:.3f}, R\u00b2={pt_stats['r_squared']:.3f}")

    print("Computing rush premium...")
    rush_premium = compute_rush_premium(df)
    if rush_premium["controlled_mean_ratio"] is not None:
        print(f"  Controlled ratio: {rush_premium['controlled_mean_ratio']:.3f}x")
    print(f"  Marginal ratio: {rush_premium['marginal_ratio']:.3f}x")
    for pt, pt_info in sorted(rush_premium.get("by_part_type", {}).items()):
        print(f"  {pt}: {pt_info['ratio']:.2f}x (n_rush={pt_info['n_rush']})")
    for mat, mat_info in sorted(rush_premium.get("by_material", {}).items()):
        print(f"  {mat}: {mat_info['ratio']:.2f}x (n_rush={mat_info['n_rush']})")

    print("Computing lead time analysis...")
    lead_time = compute_lead_time_analysis(df)
    print(f"  Correlation with unit price: {lead_time['correlation_with_unit_price']:.4f}")

    print("Computing confounding analysis...")
    confounding = compute_confounding_analysis(df)
    for pair, result in confounding["chi_squared_tests"].items():
        sig = "SIGNIFICANT" if result["significant_at_005"] else "not significant"
        print(f"  {pair}: p={result['p_value']:.4f} ({sig})")
    prop = confounding["propensity_classifier"]
    print(
        f"  Propensity accuracy: {prop['accuracy']:.3f} (baseline: {prop['random_baseline']:.3f})"
    )
    if prop["non_random_assignment"]:
        print("  => Non-random estimator assignment detected")
    else:
        print("  => Estimator assignment appears random")

    # --- Assemble JSON output ---
    profile = {
        "summary": summary,
        "unit_price_analysis": unit_price_analysis,
        "volume_discount": volume_discount,
        "rush_premium": rush_premium,
        "lead_time": lead_time,
        "missing_values": missing_report,
        "confounding": confounding,
    }

    json_path = results_dir / "data_profile.json"
    with open(json_path, "w") as f:
        json.dump(profile, f, indent=2, default=str)
    print(f"\nResults written to {json_path}")

    # --- Generate plots ---
    print("Generating plots...")
    plot_price_distribution(df, figures_dir)
    plot_unit_price_by_category(df, figures_dir)
    plot_volume_discount(df, volume_discount, figures_dir)
    plot_rush_premium(df, rush_premium, figures_dir)
    plot_lead_time(df, figures_dir)
    plot_confounding_heatmap(df, figures_dir)
    plot_missing_values(df, missing_report, figures_dir)
    print(f"Figures written to {figures_dir}/")

    print("\nEDA complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on historical quotes data")
    parser.add_argument("--data", required=True, help="Path to the CSV data file")
    parser.add_argument(
        "--output",
        required=True,
        help="Base output directory (figures/ and results/ subdirs will be created)",
    )
    args = parser.parse_args()
    main(args.data, args.output)
