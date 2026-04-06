"""Evaluate and compare trained models.

Loads CV results, generates comparison tables, runs significance tests
(Wilcoxon signed-rank), and computes feature importance / SHAP.

Usage:
    python scripts/evaluate.py --results outputs/results/cv_fold_results.csv --output outputs/
    python scripts/evaluate.py --results outputs/results/cv_fold_results.csv --compare M1,M2
"""

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="muted")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_fold_results(path: str) -> pd.DataFrame:
    """Load per-fold CV results CSV."""
    return pd.read_csv(path)


def compute_comparison_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std metrics per model from fold-level results."""
    agg = (
        fold_df.groupby("model")
        .agg(
            MAPE_mean=("MAPE", "mean"),
            MAPE_std=("MAPE", "std"),
            MedAPE_mean=("MedAPE", "mean"),
            MedAPE_std=("MedAPE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .reset_index()
    )
    return agg.sort_values("MAPE_mean").reset_index(drop=True)


def run_pairwise_significance(
    fold_df: pd.DataFrame, pairs: list[tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Run Wilcoxon signed-rank tests on per-fold MAPE.

    Tests whether the difference in MAPE between two models is
    statistically significant across CV folds.

    Args:
        fold_df: DataFrame with columns: model, fold, MAPE.
        pairs: Optional list of (model_a, model_b) tuples. If None,
            tests all pairs.

    Returns:
        DataFrame with columns: model_a, model_b, mape_a, mape_b,
        diff, p_value, significant.
    """
    models = sorted(fold_df["model"].unique())

    if pairs is None:
        pairs = list(combinations(models, 2))

    results = []
    for model_a, model_b in pairs:
        mape_a = fold_df[fold_df["model"] == model_a].sort_values("fold")["MAPE"].values
        mape_b = fold_df[fold_df["model"] == model_b].sort_values("fold")["MAPE"].values

        if len(mape_a) != len(mape_b) or len(mape_a) < 3:
            logger.warning("Skipping %s vs %s: insufficient matched folds", model_a, model_b)
            continue

        diff = mape_a - mape_b
        # Wilcoxon requires non-zero differences
        if np.all(diff == 0):
            p_value = 1.0
        else:
            try:
                _, p_value = wilcoxon(mape_a, mape_b)
            except ValueError:
                p_value = 1.0

        results.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "mape_a": float(np.mean(mape_a)),
                "mape_b": float(np.mean(mape_b)),
                "diff": float(np.mean(diff)),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }
        )

    return pd.DataFrame(results)


def resolve_hypotheses(comparison: pd.DataFrame, sig_tests: pd.DataFrame) -> pd.DataFrame:
    """Map model comparisons to the hypotheses from PLAN.md.

    Returns a table showing which hypothesis each comparison tests,
    which model won, and whether the result is significant.
    """
    hypotheses = [
        ("M0", "M1", "How much does ML add over a simple formula?"),
        ("M1", "M2", "Is pricing additive or multiplicative?"),
        ("M2", "M3a", "Does explicit unit-price decomposition help?"),
        ("M3a", "M3b", "Do discount curves differ by part type?"),
        ("M2", "M4", "Do interaction terms add lift?"),
        ("M5", "M6", "RF vs XGBoost stability (overfitting check)"),
        ("M6", "M6b", "Does lead time contribute signal?"),
        ("M6", "M7", "Raw vs log target for trees"),
        ("M6", "M7b", "XGBoost vs LightGBM on small data"),
        ("M6", "M8", "Is estimator bias additive or structural?"),
        ("M6", "M9", "Does including estimator help or hurt?"),
    ]

    # Build lookup from comparison table
    mape_lookup = dict(zip(comparison["model"], comparison["MAPE_mean"]))

    rows = []
    for model_a, model_b, question in hypotheses:
        mape_a = mape_lookup.get(model_a)
        mape_b = mape_lookup.get(model_b)

        if mape_a is None or mape_b is None:
            continue

        winner = model_a if mape_a < mape_b else model_b

        # Find significance
        sig_row = sig_tests[
            ((sig_tests["model_a"] == model_a) & (sig_tests["model_b"] == model_b))
            | ((sig_tests["model_a"] == model_b) & (sig_tests["model_b"] == model_a))
        ]
        p_value = float(sig_row["p_value"].values[0]) if len(sig_row) > 0 else None
        significant = bool(sig_row["significant"].values[0]) if len(sig_row) > 0 else None

        rows.append(
            {
                "hypothesis": question,
                "model_a": model_a,
                "model_b": model_b,
                "mape_a": mape_a,
                "mape_b": mape_b,
                "winner": winner,
                "p_value": p_value,
                "significant": significant,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_mape_comparison(comparison: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of MAPE across all models with error bars."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = comparison["model"].values
    mapes = comparison["MAPE_mean"].values
    stds = comparison["MAPE_std"].values

    bars = ax.bar(
        range(len(models)),
        mapes,
        yerr=stds,
        capsize=4,
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Model Comparison: Mean Absolute Percentage Error (5-fold CV)")

    # Add value labels
    for bar, mape in zip(bars, mapes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{mape:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "eval_mape_comparison.png", dpi=150)
    plt.close(fig)


def plot_hypothesis_resolution(hypotheses_df: pd.DataFrame, output_dir: Path) -> None:
    """Table-style plot of hypothesis resolution results."""
    fig, ax = plt.subplots(figsize=(14, max(4, len(hypotheses_df) * 0.6)))
    ax.axis("off")

    table_data = []
    for _, row in hypotheses_df.iterrows():
        sig_str = "Yes" if row.get("significant") else "No"
        if row.get("p_value") is not None:
            sig_str += f" (p={row['p_value']:.3f})"
        table_data.append(
            [
                row["hypothesis"],
                f"{row['model_a']} ({row['mape_a']:.1f}%)",
                f"{row['model_b']} ({row['mape_b']:.1f}%)",
                row["winner"],
                sig_str,
            ]
        )

    table = ax.table(
        cellText=table_data,
        colLabels=["Hypothesis", "Model A (MAPE)", "Model B (MAPE)", "Winner", "Significant?"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(5)))
    table.scale(1, 1.5)

    fig.tight_layout()
    fig.savefig(output_dir / "eval_hypothesis_resolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    results_path: str,
    output_path: str,
    compare_pairs: list[tuple[str, str]] | None = None,
) -> None:
    """Run evaluation pipeline."""
    results_path = Path(results_path)
    output_path = Path(output_path)
    results_dir = output_path / "results"
    figures_dir = output_path / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load fold results
    print("Loading fold results...")
    fold_df = load_fold_results(results_path)
    print(f"  {len(fold_df)} fold results for {fold_df['model'].nunique()} models")

    # Comparison table
    print("\nComputing comparison table...")
    comparison = compute_comparison_table(fold_df)
    csv_path = results_dir / "model_comparison.csv"
    comparison.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved to {csv_path}")
    print(comparison.to_string(index=False))

    # Significance tests
    print("\nRunning pairwise significance tests...")
    sig_tests = run_pairwise_significance(fold_df, pairs=compare_pairs)
    sig_path = results_dir / "significance_tests.csv"
    sig_tests.to_csv(sig_path, index=False, float_format="%.4f")
    print(f"  Saved to {sig_path}")

    significant_pairs = sig_tests[sig_tests["significant"]]
    print(f"  {len(significant_pairs)} significant differences found")
    if len(significant_pairs) > 0:
        for _, row in significant_pairs.iterrows():
            print(
                f"    {row['model_a']} vs {row['model_b']}: "
                f"p={row['p_value']:.4f}, diff={row['diff']:+.2f}%"
            )

    # Hypothesis resolution
    print("\nResolving hypotheses...")
    hypotheses = resolve_hypotheses(comparison, sig_tests)
    hyp_path = results_dir / "hypothesis_resolution.csv"
    hypotheses.to_csv(hyp_path, index=False, float_format="%.4f")
    print(f"  Saved to {hyp_path}")
    for _, row in hypotheses.iterrows():
        sig = "*" if row.get("significant") else ""
        print(f"  {row['hypothesis']}: {row['winner']} wins{sig}")

    # Plots
    print("\nGenerating plots...")
    plot_mape_comparison(comparison, figures_dir)
    plot_hypothesis_resolution(hypotheses, figures_dir)
    print(f"  Figures saved to {figures_dir}/")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to cv_fold_results.csv",
    )
    parser.add_argument("--output", required=True, help="Base output directory")
    parser.add_argument(
        "--compare",
        default=None,
        help="Comma-separated model pairs to compare (e.g., 'M1,M2')",
    )
    args = parser.parse_args()

    compare_pairs = None
    if args.compare:
        parts = args.compare.split(",")
        if len(parts) == 2:
            compare_pairs = [(parts[0].strip(), parts[1].strip())]

    main(args.results, args.output, compare_pairs)
