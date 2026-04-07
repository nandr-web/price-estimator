"""Start the FastAPI server with trained models loaded.

Usage:
    python scripts/serve.py --model outputs/models/M6.joblib --port 8000
    python scripts/serve.py --models-dir outputs/models/ --port 8000
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from price_estimator.api import app, set_models
from price_estimator.data import load_data
from price_estimator.predict import TrainingBounds, load_prediction_bands

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(
    model_path: str | None = None,
    models_dir: str | None = None,
    data_path: str | None = None,
    bands_path: str | None = None,
    port: int = 8000,
) -> None:
    """Load models and start the API server.

    Args:
        model_path: Path to a single serialized model.
        models_dir: Path to directory of serialized models (loads all .joblib files).
        data_path: Path to training data CSV (for OOD bounds). Optional.
        bands_path: Path to prediction_bands.json. Optional.
        port: Port to serve on.
    """
    models = {}

    if model_path:
        path = Path(model_path)
        model = joblib.load(path)
        models[model.name] = model
        print(f"Loaded model: {model.name} from {path}")

    if models_dir:
        dir_path = Path(models_dir)
        for path in sorted(dir_path.glob("*.joblib")):
            model = joblib.load(path)
            models[model.name] = model
            print(f"Loaded model: {model.name} from {path}")

    if not models:
        print("ERROR: No models loaded. Use --model or --models-dir.")
        sys.exit(1)

    # Compute training bounds for OOD detection
    bounds = None
    if data_path:
        df = load_data(data_path)
        bounds = TrainingBounds.from_dataframe(df)
        print(f"Training bounds computed from {len(df)} rows")

    # Load empirical prediction bands
    bands = None
    if bands_path:
        bands = load_prediction_bands(bands_path)
        print(f"Loaded prediction bands for {len(bands)} model(s)")

    set_models(models, bounds, prediction_bands=bands)
    print(f"\nStarting server on port {port} with {len(models)} model(s)...")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Price Estimator API")
    parser.add_argument("--model", default=None, help="Path to a single .joblib model")
    parser.add_argument("--models-dir", default=None, help="Directory of .joblib models")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to training CSV (for OOD bounds)",
    )
    parser.add_argument(
        "--bands",
        default=None,
        help="Path to prediction_bands.json (for typical range)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()
    main(args.model, args.models_dir, args.data, args.bands, args.port)
