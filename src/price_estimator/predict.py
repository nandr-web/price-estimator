"""Prediction with confidence bands, SHAP explanations, and OOD detection.

Provides the prediction interface that combines multiple models, generates
prediction intervals, and flags out-of-distribution inputs.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from price_estimator.features import build_feature_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training data bounds (for OOD detection)
# ---------------------------------------------------------------------------


class TrainingBounds:
    """Stores feature ranges from training data for OOD detection.

    Attributes:
        quantity_range: (min, max) of Quantity in training data.
        lead_time_range: (min, max) of LeadTimeWeeks.
        known_materials: Set of known material values.
        known_processes: Set of known process values.
        known_part_types: Set of known PartDescription values.
        known_estimators: Set of known estimator values.
    """

    def __init__(self):
        self.quantity_range: tuple[int, int] = (1, 100)
        self.lead_time_range: tuple[int, int] = (2, 12)
        self.known_materials: set[str] = set()
        self.known_processes: set[str] = set()
        self.known_part_types: set[str] = set()
        self.known_estimators: set[str] = set()

    def to_json(self) -> dict:
        """Serialize bounds to a JSON-compatible dict for S3 storage."""
        return {
            "quantity_range": list(self.quantity_range),
            "lead_time_range": list(self.lead_time_range),
            "known_materials": sorted(self.known_materials),
            "known_processes": sorted(self.known_processes),
            "known_part_types": sorted(self.known_part_types),
            "known_estimators": sorted(self.known_estimators),
        }

    @classmethod
    def from_json(cls, data: dict) -> "TrainingBounds":
        """Deserialize bounds from a JSON dict (e.g. loaded from S3)."""
        bounds = cls()
        bounds.quantity_range = tuple(data["quantity_range"])
        bounds.lead_time_range = tuple(data["lead_time_range"])
        bounds.known_materials = set(data["known_materials"])
        bounds.known_processes = set(data["known_processes"])
        bounds.known_part_types = set(data["known_part_types"])
        bounds.known_estimators = set(data["known_estimators"])
        return bounds

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TrainingBounds":
        """Compute bounds from training data.

        Args:
            df: Training DataFrame.

        Returns:
            TrainingBounds instance.
        """
        bounds = cls()
        bounds.quantity_range = (int(df["Quantity"].min()), int(df["Quantity"].max()))
        bounds.lead_time_range = (
            int(df["LeadTimeWeeks"].min()),
            int(df["LeadTimeWeeks"].max()),
        )
        bounds.known_materials = set(df["Material"].dropna().unique())
        bounds.known_processes = set(df["Process"].dropna().unique())
        bounds.known_part_types = set(df["PartDescription"].unique())
        bounds.known_estimators = set(df["Estimator"].unique())
        return bounds


def detect_ood(df: pd.DataFrame, bounds: TrainingBounds) -> list[dict]:
    """Detect out-of-distribution inputs.

    Checks each row against training data bounds and returns a list of
    OOD flags with explanations.

    Args:
        df: Input DataFrame to check.
        bounds: Training data bounds.

    Returns:
        List of dicts, one per row, each containing:
            - is_ood: bool
            - reasons: list of strings explaining why
    """
    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        reasons = []

        # Quantity range
        qty = row["Quantity"]
        if qty < bounds.quantity_range[0] or qty > bounds.quantity_range[1]:
            reasons.append(
                f"Quantity {qty} outside training range "
                f"[{bounds.quantity_range[0]}, {bounds.quantity_range[1]}]"
            )

        # Lead time range
        lt = row["LeadTimeWeeks"]
        if lt < bounds.lead_time_range[0] or lt > bounds.lead_time_range[1]:
            reasons.append(
                f"LeadTimeWeeks {lt} outside training range "
                f"[{bounds.lead_time_range[0]}, {bounds.lead_time_range[1]}]"
            )

        # Unknown material
        mat = row.get("Material")
        if pd.notna(mat) and mat not in bounds.known_materials:
            reasons.append(f"Unknown material: {mat}")

        # Missing material
        if pd.isna(mat):
            reasons.append("Missing material")

        # Unknown process
        proc = row.get("Process")
        if pd.notna(proc) and proc not in bounds.known_processes:
            reasons.append(f"Unknown process: {proc}")

        # Missing process
        if pd.isna(proc):
            reasons.append("Missing process")

        # Unknown part description
        desc = row.get("PartDescription", "")
        if desc not in bounds.known_part_types:
            reasons.append(f"Unknown part description: {desc}")

        results.append({"is_ood": len(reasons) > 0, "reasons": reasons})

    return results


def compute_model_disagreement(
    predictions: dict[str, np.ndarray],
) -> dict:
    """Compute multi-model disagreement statistics.

    Args:
        predictions: Dict mapping model name to prediction array.
            All arrays must have the same length.

    Returns:
        Dictionary with:
            - min_pred: array of per-row minimums across models
            - max_pred: array of per-row maximums across models
            - median_pred: array of per-row medians
            - spread_pct: array of (max-min)/median as percentage
            - mean_spread_pct: scalar mean of spread_pct
            - flagged_indices: indices where spread > 20%
    """
    pred_matrix = np.column_stack(list(predictions.values()))

    min_pred = pred_matrix.min(axis=1)
    max_pred = pred_matrix.max(axis=1)
    median_pred = np.median(pred_matrix, axis=1)

    spread_pct = (max_pred - min_pred) / median_pred * 100

    return {
        "min_pred": min_pred,
        "max_pred": max_pred,
        "median_pred": median_pred,
        "spread_pct": spread_pct,
        "mean_spread_pct": float(np.mean(spread_pct)),
        "flagged_indices": np.where(spread_pct > 20)[0],
    }


# ---------------------------------------------------------------------------
# Top-tier model consensus
# ---------------------------------------------------------------------------

# Top-tier selection criteria:
# 1. MAPE < 14% and perfect economic coherence (10/10)
# 2. Structural diversity: at least one linear AND one tree model
# 3. Segment-weighted: include models that perform well on expensive jobs
#    (where underquoting hurts most), not just best overall MAPE
#
# Linear family: M2 (Ridge log-linear), M3a/M3b (two-stage), M4 (Lasso)
# Tree family: M7 (XGBoost log-target) — best tree model at 12.7% MAPE
#   with perfect economic coherence
TOP_TIER_LINEAR = {"M2", "M3a", "M3b", "M4"}
TOP_TIER_TREE = {"M7"}
TOP_TIER_MODELS = TOP_TIER_LINEAR | TOP_TIER_TREE


def compute_recommendation(
    predictions: dict[str, float],
    band: dict | None = None,
) -> dict:
    """Compute recommended quote values from top-tier model consensus.

    Uses the median of top-tier models as the best estimate of true cost,
    then shifts using the empirical error band to produce conservative
    (protects margin) and aggressive (wins bids) values.

    The top tier includes both linear (M2, M3a, M3b, M4) and tree (M7)
    models for structural diversity. When these families disagree, it's
    a signal that the job has characteristics one family handles better.

    Win bid = shifted 75% toward lower band edge (competitive price)
    Protect margin = shifted 75% toward upper band edge (safer margin)

    Args:
        predictions: Dict mapping model name to prediction value.
        band: Empirical band dict from compute_empirical_bands().
            If None, only the estimate is returned.

    Returns:
        Dictionary with:
            - estimate: median of top-tier models (best guess of true cost)
            - win_bid: lower quote, more competitive
            - protect_margin: higher quote, safer margin
            - top_tier_models: which models contributed
            - top_tier_spread_pct: spread among top-tier models as %
            - family_divergence: dict with linear/tree medians and spread,
              or None if both families aren't represented
    """
    top_preds = {m: p for m, p in predictions.items() if m in TOP_TIER_MODELS}

    # Fall back to all models if no top-tier models are available
    if not top_preds:
        top_preds = predictions

    vals = list(top_preds.values())
    estimate = float(np.median(vals))

    result = {
        "estimate": estimate,
        "win_bid": None,
        "protect_margin": None,
        "top_tier_models": sorted(top_preds.keys()),
        "top_tier_spread_pct": (
            float((max(vals) - min(vals)) / estimate * 100) if estimate > 0 else 0.0
        ),
        "family_divergence": None,
    }

    # Check linear vs tree family divergence
    linear_preds = [p for m, p in top_preds.items() if m in TOP_TIER_LINEAR]
    tree_preds = [p for m, p in top_preds.items() if m in TOP_TIER_TREE]

    if linear_preds and tree_preds:
        linear_med = float(np.median(linear_preds))
        tree_med = float(np.median(tree_preds))
        midpoint = (linear_med + tree_med) / 2
        divergence_pct = abs(linear_med - tree_med) / midpoint * 100

        result["family_divergence"] = {
            "linear_median": round(linear_med, 2),
            "tree_median": round(tree_med, 2),
            "divergence_pct": round(divergence_pct, 1),
        }

    if band:
        # Band has lower_pct (e.g. -14.5%) and upper_pct (e.g. +17.9%)
        # These are signed errors: positive = model overestimated.
        #
        # Win bid: shift lower (underestimate direction).
        # Protect margin: shift higher (overestimate direction).
        # Factor 0.75: at 0.25 PM underquoted 28% of jobs ($197K exposure);
        # at 0.75 exposure drops to 12% ($112K), concentrated in genuine
        # outliers rather than routine jobs.
        margin_shift = band["upper_pct"] * 0.75 / 100
        bid_shift = band["lower_pct"] * 0.75 / 100

        result["protect_margin"] = round(estimate * (1 + margin_shift), 2)
        result["win_bid"] = round(estimate * (1 + bid_shift), 2)

    result["estimate"] = round(estimate, 2)
    return result


# ---------------------------------------------------------------------------
# Empirical prediction bands
# ---------------------------------------------------------------------------


def compute_empirical_bands(
    actuals: np.ndarray,
    predictions: np.ndarray,
    coverage: float = 0.80,
) -> dict:
    """Compute empirical prediction bands from out-of-fold residuals.

    Uses signed percentage errors from CV predictions to derive lower
    and upper multipliers. Applying these to a new prediction gives a
    "typical range" — e.g., 80% of similar historical jobs fell within
    this band.

    Args:
        actuals: Actual prices from OOF predictions.
        predictions: Model predictions from OOF predictions.
        coverage: Desired coverage probability (default 0.80).

    Returns:
        Dictionary with:
            - coverage: the coverage level
            - lower_pct: lower percentile of signed errors (e.g. -12.5%)
            - upper_pct: upper percentile of signed errors (e.g. +15.3%)
            - median_abs_error_pct: median absolute error percentage
    """
    # Signed percentage errors: positive = model overestimated
    signed_pct = (predictions - actuals) / actuals * 100
    abs_pct = np.abs(signed_pct)

    tail = (1 - coverage) / 2
    lower_pct = float(np.percentile(signed_pct, tail * 100))
    upper_pct = float(np.percentile(signed_pct, (1 - tail) * 100))

    return {
        "coverage": coverage,
        "lower_pct": lower_pct,
        "upper_pct": upper_pct,
        "median_abs_error_pct": float(np.median(abs_pct)),
    }


def save_prediction_bands(
    bands_by_model: dict[str, dict],
    path: str | Path,
) -> None:
    """Save empirical prediction bands to a JSON file.

    Args:
        bands_by_model: Dict mapping model name to band dict from
            compute_empirical_bands().
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(bands_by_model, f, indent=2)
    logger.info("Saved prediction bands to %s", path)


def load_prediction_bands(path: str | Path) -> dict[str, dict]:
    """Load empirical prediction bands from a JSON file.

    Args:
        path: Path to the bands JSON file.

    Returns:
        Dict mapping model name to band dict.
    """
    with open(path) as f:
        return json.load(f)


def apply_prediction_band(
    prediction: float,
    band: dict,
) -> tuple[float, float]:
    """Apply empirical band to a point prediction.

    Args:
        prediction: The model's point prediction.
        band: Band dict from compute_empirical_bands().

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    # The band percentiles represent model error direction.
    # If model overestimates by 15% at P90, the actual was 15% lower.
    # So lower bound = prediction / (1 + upper_pct/100)
    # and upper bound = prediction / (1 + lower_pct/100)
    lower = prediction / (1 + band["upper_pct"] / 100)
    upper = prediction / (1 + band["lower_pct"] / 100)
    return (max(lower, 0), upper)


def compute_shap_explanation(model, df: pd.DataFrame) -> dict:
    """Compute SHAP values for a prediction.

    Args:
        model: A trained model with a sklearn-compatible predict method.
            Must have a .model attribute (the underlying estimator) and
            ._feature_cols attribute.
        df: Single-row or multi-row DataFrame to explain.

    Returns:
        Dictionary with:
            - shap_values: array of shape (n_samples, n_features)
            - feature_names: list of feature names
            - base_value: expected value (model baseline)
    """
    import shap

    X, _ = build_feature_matrix(df, encoding="label")
    X = X.fillna(-1)
    if hasattr(model, "_feature_cols"):
        X = X.reindex(columns=model._feature_cols, fill_value=-1)

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X.values)

    return {
        "shap_values": shap_values,
        "feature_names": list(X.columns),
        "base_value": float(explainer.expected_value),
    }


def format_shap_explanation(explanation: dict, top_n: int = 5) -> str:
    """Format SHAP values as a human-readable string.

    Shows the top N features by absolute SHAP value for the first
    prediction in the explanation.

    Args:
        explanation: Output from compute_shap_explanation().
        top_n: Number of top features to show.

    Returns:
        Formatted string.
    """
    shap_vals = explanation["shap_values"]
    if shap_vals.ndim > 1:
        shap_vals = shap_vals[0]

    names = explanation["feature_names"]
    base = explanation["base_value"]

    # Sort by absolute value
    indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]

    lines = [f"Base value: ${base:,.2f}"]
    for idx in indices:
        name = names[idx]
        val = shap_vals[idx]
        sign = "+" if val >= 0 else ""
        lines.append(f"  {name}: {sign}${val:,.2f}")

    prediction = base + np.sum(shap_vals)
    lines.append(f"Prediction: ${prediction:,.2f}")

    return "\n".join(lines)
