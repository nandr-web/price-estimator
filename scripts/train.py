"""Train all models with cross-validation and save results.

Trains M0 through M9 (or a subset), runs 5-fold CV, saves trained
models to outputs/models/ and CV results to outputs/results/.

Usage:
    python scripts/train.py --data resources/aora_historical_quotes.csv --output outputs/
    python scripts/train.py --data resources/aora_historical_quotes.csv --model M6 --output outputs
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from price_estimator.data import load_data, validate
from price_estimator.models import get_all_models, get_model_by_name, results_to_dataframe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(data_path: str, output_path: str, model_names: list[str] | None = None) -> None:
    """Train models and save results.

    Args:
        data_path: Path to the CSV data file.
        output_path: Base output directory.
        model_names: Optional list of model names to train. If None, trains all.
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    models_dir = output_path / "models"
    results_dir = output_path / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_data(data_path)
    validate(df)
    print(f"  {len(df)} rows loaded")

    # Get models
    if model_names:
        models = [get_model_by_name(name) for name in model_names]
    else:
        models = get_all_models()

    print(f"\nTraining {len(models)} models with 5-fold CV...")
    print("=" * 60)

    all_results = []
    for model in models:
        print(f"\n--- {model.name} ---")
        try:
            cv_result = model.cross_validate(df)
            all_results.append(cv_result)

            mape = cv_result.mean_metrics["MAPE"]
            mape_std = cv_result.std_metrics["MAPE"]
            rmse = cv_result.mean_metrics["RMSE"]
            r2 = cv_result.mean_metrics["R2"]

            print(f"  MAPE:    {mape:.2f}% (+/- {mape_std:.2f}%)")
            print(f"  RMSE:    ${rmse:,.2f}")
            print(f"  R²:      {r2:.4f}")

            # Save trained model (retrain on full data)
            model.fit(df)
            model_path = models_dir / f"{model.name}.joblib"
            joblib.dump(model, model_path)
            print(f"  Saved:   {model_path}")

        except Exception:
            logger.exception("Failed to train %s", model.name)

    # Save comparison table
    if all_results:
        comparison = results_to_dataframe(all_results)
        csv_path = results_dir / "cv_results.csv"
        comparison.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"\n{'=' * 60}")
        print(f"CV results saved to {csv_path}")
        print("\nModel comparison (sorted by MAPE):")
        print(comparison.to_string(index=False))

        # Also save per-fold results for significance testing
        fold_rows = []
        for result in all_results:
            for fold_idx, metrics in enumerate(result.fold_metrics):
                row = {"model": result.model_name, "fold": fold_idx}
                row.update(metrics)
                fold_rows.append(row)
        fold_df = pd.DataFrame(fold_rows)
        fold_path = results_dir / "cv_fold_results.csv"
        fold_df.to_csv(fold_path, index=False, float_format="%.4f")
        print(f"Per-fold results saved to {fold_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train price estimator models")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--output", required=True, help="Base output directory")
    parser.add_argument(
        "--model",
        nargs="*",
        default=None,
        help="Model name(s) to train (default: all)",
    )
    args = parser.parse_args()
    main(args.data, args.output, args.model)
