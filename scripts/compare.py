"""Run the 7-lens model comparison framework.

Loads trained models and data, runs all comparison lenses, and outputs
a structured report (JSON + text summary + figures).

Usage:
    python scripts/compare.py --data resources/aora_historical_quotes.csv \
        --models outputs/models/ --output outputs/
    python scripts/compare.py --data resources/aora_historical_quotes.csv \
        --models outputs/models/ --model M0 M2 M6
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="muted")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from price_estimator.comparison import (  # noqa: E402
    boundary_behavior,
    calibration_bias,
    complexity_interpretability,
    economic_coherence,
    error_profile,
    format_scorecard_text,
    generate_scorecard,
    segment_fairness,
    stability_robustness,
)
from price_estimator.data import load_data, validate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_models(model_dir: Path, model_names: list[str] | None = None) -> dict:
    """Load serialized models from the models directory."""
    models = {}
    for path in sorted(model_dir.glob("*.joblib")):
        name = path.stem
        if model_names and name not in model_names:
            continue
        try:
            models[name] = joblib.load(path)
            logger.info("Loaded %s from %s", name, path)
        except Exception:
            logger.exception("Failed to load %s", name)
    return models


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_error_profiles(error_results: dict, output_dir: Path) -> None:
    """Histogram of APE distributions per model."""
    n_models = len(error_results)
    if n_models == 0:
        return

    cols = min(4, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for idx, (name, profile) in enumerate(sorted(error_results.items())):
        ax = axes[idx // cols][idx % cols]
        ape = profile["ape_values"]
        ax.hist(ape, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        mape_label = f"MAPE={profile['mape']:.1f}%"
        p90_label = f"P90={profile['p90_ape']:.1f}%"
        ax.axvline(profile["mape"], color="red", linestyle="--", label=mape_label)
        ax.axvline(profile["p90_ape"], color="orange", linestyle=":", label=p90_label)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("APE (%)")
        ax.set_xlim(0, min(100, profile["p95_ape"] * 1.5))
        ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n_models, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Error Distribution by Model", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "compare_error_profiles.png", dpi=150)
    plt.close(fig)


def plot_segment_heatmap(segment_results: dict, output_dir: Path) -> None:
    """Heatmap of MAPE by segment for each model."""
    # Use by_material segment as the representative heatmap
    models = sorted(segment_results.keys())
    segment_key = "by_material"

    all_cats = set()
    for m in models:
        all_cats.update(segment_results[m].get(segment_key, {}).keys())
    categories = sorted(all_cats)

    if not categories:
        return

    data = np.full((len(models), len(categories)), np.nan)
    for i, m in enumerate(models):
        seg = segment_results[m].get(segment_key, {})
        for j, cat in enumerate(categories):
            if cat in seg:
                data[i, j] = seg[cat]["mape"]

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.5), max(4, len(models) * 0.5)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=categories,
        yticklabels=models,
        ax=ax,
        cbar_kws={"label": "MAPE (%)"},
    )
    ax.set_title("Segment Fairness: MAPE by Material")
    fig.tight_layout()
    fig.savefig(output_dir / "compare_segment_fairness.png", dpi=150)
    plt.close(fig)


def plot_calibration(calibration_results: dict, output_dir: Path) -> None:
    """Bar chart of mean signed error per model."""
    models = sorted(calibration_results.keys())
    signed_errors = [calibration_results[m]["mean_signed_error_pct"] for m in models]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.8), 5))
    colors = ["#e74c3c" if e < -3 else "#2ecc71" if e > 3 else "#3498db" for e in signed_errors]
    bars = ax.bar(models, signed_errors, color=colors, edgecolor="black", alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(3, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(-3, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Mean Signed Error (%)")
    ax.set_title("Calibration: Bias Direction\n(positive = conservative, negative = aggressive)")
    plt.xticks(rotation=45, ha="right")

    for bar, val in zip(bars, signed_errors):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + (0.3 if y >= 0 else -0.8),
            f"{val:+.1f}%",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "compare_calibration.png", dpi=150)
    plt.close(fig)


def plot_stability(stability_results: dict, output_dir: Path) -> None:
    """Box plot of per-fold MAPEs."""
    models = sorted(stability_results.keys())
    fold_data = [stability_results[m]["fold_mapes"] for m in models]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.8), 5))
    bp = ax.boxplot(fold_data, labels=models, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)

    ax.set_ylabel("MAPE (%)")
    ax.set_title("Stability: Fold-to-Fold MAPE Variation")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "compare_stability.png", dpi=150)
    plt.close(fig)


def plot_coherence_summary(coherence_results: dict, output_dir: Path) -> None:
    """Bar chart of economic coherence pass rates."""
    models = sorted(coherence_results.keys())
    pass_rates = [coherence_results[m]["pass_rate"] for m in models]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.8), 5))
    colors = ["#2ecc71" if r == 100 else "#e67e22" if r >= 75 else "#e74c3c" for r in pass_rates]
    ax.bar(models, pass_rates, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 110)
    ax.axhline(100, color="green", linestyle="--", alpha=0.3)
    ax.set_title(
        "Economic Coherence: Pass Rate\n(material ordering, qty discount, rush, complexity)"
    )
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "compare_coherence.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON serializer
# ---------------------------------------------------------------------------


def _make_serializable(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    data_path: str,
    model_dir: str,
    output_path: str,
    model_names: list[str] | None = None,
    n_bootstrap: int = 20,
) -> None:
    """Run the 7-lens comparison framework."""
    data_path = Path(data_path)
    model_dir = Path(model_dir)
    output_path = Path(output_path)
    results_dir = output_path / "results"
    figures_dir = output_path / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_data(data_path)
    warnings = validate(df)
    for w in warnings:
        print(f"  Warning: {w}")
    print(f"  {len(df)} rows loaded")

    # Load models
    print(f"\nLoading models from {model_dir}...")
    models = load_models(model_dir, model_names)
    if not models:
        print("ERROR: No models found. Run scripts/train.py first.")
        sys.exit(1)
    print(f"  Loaded: {', '.join(sorted(models.keys()))}")

    # Run all 7 lenses
    print("\n--- Lens 1: Error Profile ---")
    err = error_profile(models, df)
    for name in sorted(err.keys()):
        e = err[name]
        print(
            f"  {name}: MAPE={e['mape']:.1f}%, MedAPE={e['median_ape']:.1f}%, "
            f"P90={e['p90_ape']:.1f}%, Max={e['max_ape']:.1f}%, "
            f"<10%={e['pct_under_10']:.0f}%, >50%={e['pct_over_50']:.0f}%"
        )

    print("\n--- Lens 2: Segment Fairness ---")
    seg = segment_fairness(models, df)
    for name in sorted(seg.keys()):
        worst_seg = ""
        worst_mape = 0
        for seg_name, cats in seg[name].items():
            for cat, stats in cats.items():
                if stats["count"] >= 10 and stats["mape"] > worst_mape:
                    worst_mape = stats["mape"]
                    worst_seg = f"{seg_name}/{cat}"
        print(f"  {name}: worst segment = {worst_seg} ({worst_mape:.1f}% MAPE)")

    print("\n--- Lens 3: Economic Coherence ---")
    coh = economic_coherence(models, df)
    for name in sorted(coh.keys()):
        c = coh[name]
        pc, tc, pr = c["pass_count"], c["total_count"], c["pass_rate"]
        print(f"  {name}: {pc}/{tc} checks passed ({pr:.0f}%)")
        for check in c["checks"]:
            if not check["passed"]:
                details = check.get("details", [])
                if details:
                    for d in details[:3]:
                        print(f"    FAIL [{check['name']}]: {d}")

    print("\n--- Lens 4: Calibration & Bias Direction ---")
    cal = calibration_bias(models, df)
    for name in sorted(cal.keys()):
        c = cal[name]
        print(
            f"  {name}: {c['label']} (mean signed error = {c['mean_signed_error_pct']:+.1f}%, "
            f"overestimates {c['pct_overestimated']:.0f}% of quotes)"
        )

    print("\n--- Lens 5: Stability & Robustness ---")
    stab = stability_robustness(models, df, n_bootstrap=n_bootstrap)
    for name in sorted(stab.keys()):
        s = stab[name]
        print(
            f"  {name}: fold MAPE = {s['fold_mape_mean']:.1f}% ± {s['fold_mape_std']:.1f}%, "
            f"bootstrap CV = {s['bootstrap_mean_cv_pct']:.1f}%"
        )

    print("\n--- Lens 6: Boundary Behavior ---")
    bound = boundary_behavior(models, df)
    for name in sorted(bound.keys()):
        b = bound[name]
        print(f"  {name}: {b['pass_count']}/{b['total_count']} boundary tests passed")
        for t in b["tests"]:
            passed = t.get("passed", t.get("reasonable", t.get("handles_gracefully", False)))
            if not passed:
                print(f"    FAIL [{t['name']}]: {t.get('details', t.get('failure_mode', ''))}")

    print("\n--- Lens 7: Complexity & Interpretability ---")
    comp = complexity_interpretability(models, df)
    for name in sorted(comp.keys()):
        c = comp[name]
        print(
            f"  {name}: {c['interpretability_rating']} interpretability, "
            f"{c['effective_params']} effective params, "
            f"{c['train_seconds']:.2f}s training time"
        )

    # Generate scorecard
    print("\n" + "=" * 80)
    scorecard = generate_scorecard(err, seg, coh, cal, stab, bound, comp)
    scorecard_text = format_scorecard_text(scorecard)
    print(scorecard_text)

    # Save results
    report = {
        "scorecard": _make_serializable(scorecard),
        "error_profile": _make_serializable(
            {m: {k: v for k, v in d.items() if k != "ape_values"} for m, d in err.items()}
        ),
        "segment_fairness": _make_serializable(seg),
        "economic_coherence": _make_serializable(coh),
        "calibration_bias": _make_serializable(cal),
        "stability_robustness": _make_serializable(stab),
        "boundary_behavior": _make_serializable(bound),
        "complexity_interpretability": _make_serializable(comp),
    }

    json_path = results_dir / "comparison_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved report to {json_path}")

    text_path = results_dir / "comparison_scorecard.txt"
    with open(text_path, "w") as f:
        f.write(scorecard_text)
    print(f"Saved scorecard to {text_path}")

    # Generate figures
    print("\nGenerating figures...")
    plot_error_profiles(err, figures_dir)
    plot_segment_heatmap(seg, figures_dir)
    plot_calibration(cal, figures_dir)
    plot_stability(stab, figures_dir)
    plot_coherence_summary(coh, figures_dir)
    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="7-lens model comparison")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--models", required=True, help="Path to models directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", nargs="*", help="Specific model names to compare")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=20,
        help="Bootstrap resamples for stability",
    )
    args = parser.parse_args()

    main(
        data_path=args.data,
        model_dir=args.models,
        output_path=args.output,
        model_names=args.model,
        n_bootstrap=args.bootstrap,
    )
