"""Predict a single quote using one or all trained models.

Usage examples:
    # Manual input with a specific model
    python scripts/predict_one.py \
        --model M2 \
        --part "Sensor Housing - threaded" \
        --material "Inconel 718" \
        --process "5-Axis Milling" \
        --quantity 10 \
        --rush \
        --lead-time 4 \
        --estimator "Tanaka-san"

    # All models
    python scripts/predict_one.py \
        --model all \
        --part "Bracket" \
        --material "Aluminum 6061" \
        --process "3-Axis Milling" \
        --quantity 5 \
        --lead-time 6

    # Pull row from the dataset and compare actual vs predicted
    python scripts/predict_one.py --model all --csv-row 42
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from price_estimator.data import load_data  # noqa: E402
from price_estimator.predict import (  # noqa: E402
    apply_prediction_band,
    compute_recommendation,
    load_prediction_bands,
)


def load_models(model_dir: Path, model_name: str | None = None) -> dict:
    """Load one or all serialized models from the models directory.

    Args:
        model_dir: Path to directory containing .joblib files.
        model_name: If provided and not 'all', load only this model.

    Returns:
        Dict mapping model name to loaded model object.
    """
    models = {}
    for path in sorted(model_dir.glob("*.joblib")):
        name = path.stem
        if model_name and model_name.lower() != "all" and name != model_name:
            continue
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            print(f"  Warning: failed to load {name}: {e}")
    return models


ESTIMATORS = ["Sato-san", "Suzuki-san", "Tanaka-san"]


def build_input_dataframe(
    part: str,
    material: str | None,
    process: str | None,
    quantity: int,
    lead_time: int,
    rush: bool,
    estimator: str,
) -> pd.DataFrame:
    """Build a single-row DataFrame from CLI arguments."""
    return pd.DataFrame(
        [
            {
                "QuoteID": "CLI-001",
                "Date": pd.Timestamp.now(),
                "PartDescription": part,
                "Material": material,
                "Process": process,
                "Quantity": quantity,
                "LeadTimeWeeks": lead_time,
                "RushJob": rush,
                "Estimator": estimator,
                "TotalPrice_USD": 0.0,
            }
        ]
    )


def print_inputs(
    df: pd.DataFrame,
    estimator_provided: bool = True,
) -> None:
    """Print the input features in a readable format."""
    row = df.iloc[0]
    rush_str = "Yes" if row["RushJob"] else "No"
    material_str = row["Material"] if pd.notna(row["Material"]) else "(missing)"
    process_str = row["Process"] if pd.notna(row["Process"]) else "(missing)"

    print("\nInput:")
    print(f"  Part:       {row['PartDescription']}")
    print(f"  Material:   {material_str}")
    print(f"  Process:    {process_str}")
    print(f"  Quantity:   {row['Quantity']}")
    print(f"  Rush:       {rush_str}")
    print(f"  Lead Time:  {row['LeadTimeWeeks']} weeks")
    if estimator_provided:
        print(f"  Estimator:  {row['Estimator']}")
    else:
        print("  Estimator:  (not provided — using M9 debiased + per-estimator range)")


def run_predictions(models: dict, df: pd.DataFrame) -> dict[str, float]:
    """Run prediction on df for each model, returning name -> prediction."""
    predictions = {}
    for name, model in sorted(models.items()):
        try:
            pred = model.predict(df)
            predictions[name] = float(pred[0])
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")
    return predictions


def run_predictions_no_estimator(
    models: dict,
    df_template: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run predictions without a known estimator.

    For debiased models (M9, M0): predict directly (they don't use estimator).
    For estimator-aware models: predict with each of the 3 estimators,
    report per-estimator range.

    Returns:
        Tuple of (model_predictions, estimator_range) where:
          model_predictions maps model name to its best prediction
            (M9 preferred, otherwise median across estimators)
          estimator_range maps estimator name to median prediction
            across all estimator-aware models
    """
    # Models that don't use estimator feature
    debiased_models = {"M0", "M9"}

    model_predictions = {}
    per_estimator_totals: dict[str, list[float]] = {e: [] for e in ESTIMATORS}

    for name, model in sorted(models.items()):
        if name in debiased_models:
            try:
                pred = model.predict(df_template)
                model_predictions[name] = float(pred[0])
            except Exception as e:
                print(f"  Warning: {name} failed: {e}")
        else:
            # Run with each estimator, take median as the model's prediction
            est_preds = {}
            for est in ESTIMATORS:
                df_est = df_template.copy()
                df_est["Estimator"] = est
                try:
                    pred = model.predict(df_est)
                    est_preds[est] = float(pred[0])
                    per_estimator_totals[est].append(float(pred[0]))
                except Exception as e:
                    print(f"  Warning: {name}/{est} failed: {e}")

            if est_preds:
                model_predictions[name] = float(np.median(list(est_preds.values())))

    # Estimator range: median across all estimator-aware models
    estimator_range = {}
    for est in ESTIMATORS:
        if per_estimator_totals[est]:
            estimator_range[est] = float(np.median(per_estimator_totals[est]))

    return model_predictions, estimator_range


def print_predictions(
    predictions: dict[str, float],
    actual: float | None = None,
    estimator_range: dict[str, float] | None = None,
    bands: dict[str, dict] | None = None,
) -> None:
    """Print predictions in a tabular format."""
    if not predictions:
        print("\n  No predictions available.")
        return

    # --- Recommendation (top-tier consensus) ---
    # Pick the best available band for the recommendation
    best_band = None
    if bands:
        available = [m for m in bands if m in predictions]
        if available:
            best_model = min(
                available,
                key=lambda m: bands[m]["median_abs_error_pct"],
            )
            best_band = bands[best_model]

    rec = compute_recommendation(predictions, band=best_band)

    print("\n  Recommendation (consensus of top-tier models):")
    print(f"    Estimate:       ${rec['estimate']:>12,.2f}")
    if rec["win_bid"] is not None:
        print(f"    Win bid:        ${rec['win_bid']:>12,.2f}")
        print(f"    Protect margin: ${rec['protect_margin']:>12,.2f}")
    div = rec.get("family_divergence")
    if div and div["divergence_pct"] > 10:
        print(f"\n    Linear models:  ${div['linear_median']:>12,.2f}")
        print(f"    Tree models:    ${div['tree_median']:>12,.2f}")
        print(
            f"    Divergence:     {div['divergence_pct']:.0f}%"
            " — linear and tree models disagree, review recommended"
        )

    # --- Typical range ---
    if best_band and best_model in predictions:
        pred = predictions[best_model]
        low, high = apply_prediction_band(pred, best_band)
        cov = int(best_band["coverage"] * 100)
        print(f"\n  Typical range ({cov}% of similar past jobs):")
        print(f"    ${low:>12,.2f}  —  ${high:>12,.2f}")

    # --- Model spread ---
    if len(predictions) > 1:
        vals = list(predictions.values())
        print("\n  Model range (min/max across all models):")
        print(f"    ${min(vals):>12,.2f}  —  ${max(vals):>12,.2f}")

    # --- Estimator range ---
    if estimator_range:
        print("\n  Estimator range (if quoted by each estimator):")
        for est in ESTIMATORS:
            if est in estimator_range:
                label = est.replace("-san", "")
                print(f"    {label:<12} ${estimator_range[est]:>12,.2f}")

    # --- Actual comparison ---
    if actual is not None and actual > 0:
        error_pct = abs(rec["estimate"] - actual) / actual * 100
        print()
        print(f"  Actual:  ${actual:>12,.2f}")
        print(f"  Error:   {error_pct:.1f}%")

    # --- Detailed model breakdown ---
    print("\n  All models:")
    print(f"  {'Model':<10} {'Prediction':>14}")
    print(f"  {'-' * 10} {'-' * 14}")
    for name, pred in sorted(predictions.items()):
        marker = " *" if name in rec["top_tier_models"] else ""
        print(f"  {name:<10} ${pred:>12,.2f}{marker}")
    print("  (* = top-tier model)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a single quote using trained models.")
    parser.add_argument(
        "--model",
        default="all",
        help="Model name (e.g. M2, M6) or 'all' to run all models (default: all)",
    )
    parser.add_argument(
        "--models-dir",
        default="outputs/models/",
        help="Directory containing .joblib model files (default: outputs/models/)",
    )

    # Manual input arguments
    parser.add_argument("--part", help="Part description (e.g. 'Sensor Housing - threaded')")
    parser.add_argument("--material", help="Material (e.g. 'Inconel 718')")
    parser.add_argument("--process", help="Process (e.g. '5-Axis Milling')")
    parser.add_argument("--quantity", type=int, default=10, help="Quantity (default: 10)")
    parser.add_argument("--lead-time", type=int, default=4, help="Lead time in weeks (default: 4)")
    parser.add_argument("--rush", action="store_true", help="Rush job flag")
    parser.add_argument(
        "--estimator",
        help="Estimator name. If omitted, uses M9 (debiased) as primary "
        "and reports per-estimator range from other models.",
    )

    # CSV row mode
    parser.add_argument(
        "--csv-row",
        type=int,
        help="Row index from the dataset to use as input (0-based)",
    )
    parser.add_argument(
        "--data",
        default="resources/aora_historical_quotes.csv",
        help="Path to CSV data file (default: resources/aora_historical_quotes.csv)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.csv_row is None and args.part is None:
        parser.error("Either --part or --csv-row is required")

    # Load models
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"ERROR: Models directory not found: {models_dir}")
        print("Run scripts/train.py first to train models.")
        sys.exit(1)

    models = load_models(models_dir, args.model)
    if not models:
        print(f"ERROR: No models loaded from {models_dir}")
        if args.model.lower() != "all":
            print(f"  Model '{args.model}' not found. Available .joblib files:")
            for p in sorted(models_dir.glob("*.joblib")):
                print(f"    {p.stem}")
        sys.exit(1)

    print(f"Loaded {len(models)} model(s): {', '.join(sorted(models.keys()))}")

    # Load prediction bands if available
    bands_path = Path(args.models_dir).parent / "results" / "prediction_bands.json"
    bands = None
    if bands_path.exists():
        bands = load_prediction_bands(bands_path)
    else:
        print("  (No prediction bands found — run train.py to generate)")

    actual = None
    estimator_provided = True

    if args.csv_row is not None:
        # Load from CSV — estimator is known from the data
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"ERROR: Data file not found: {data_path}")
            sys.exit(1)

        df_all = load_data(data_path)
        if args.csv_row < 0 or args.csv_row >= len(df_all):
            print(f"ERROR: Row index {args.csv_row} out of range (0-{len(df_all) - 1})")
            sys.exit(1)

        row = df_all.iloc[[args.csv_row]].copy()
        actual = float(row["TotalPrice_USD"].iloc[0])
        print(f"Using row {args.csv_row} (QuoteID: {row['QuoteID'].iloc[0]})")

        df = row.reset_index(drop=True)
    else:
        # Build from CLI arguments
        estimator_provided = args.estimator is not None
        df = build_input_dataframe(
            part=args.part,
            material=args.material,
            process=args.process,
            quantity=args.quantity,
            lead_time=args.lead_time,
            rush=args.rush,
            estimator=args.estimator or "Suzuki-san",  # placeholder for template
        )

    print_inputs(df, estimator_provided=estimator_provided)

    if estimator_provided:
        predictions = run_predictions(models, df)
        print_predictions(predictions, actual=actual, bands=bands)
    else:
        predictions, estimator_range = run_predictions_no_estimator(models, df)
        print_predictions(
            predictions,
            actual=actual,
            estimator_range=estimator_range,
            bands=bands,
        )


if __name__ == "__main__":
    main()
