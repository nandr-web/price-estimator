"""Prediction with confidence bands, SHAP explanations, and OOD detection.

Provides the prediction interface that combines multiple models, generates
prediction intervals, and flags out-of-distribution inputs.
"""

import logging

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
