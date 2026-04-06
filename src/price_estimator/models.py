"""Model definitions, training, and cross-validation for the price estimator.

Implements models M0 through M9 as defined in PLAN.md. Each model is
a self-contained class with fit/predict/cv methods. All models are
evaluated on the same seeded CV splits for fair comparison.

Model matrix:
    M0  - Lookup table / formula (no ML)
    M1  - Ridge regression (additive, raw target)
    M2  - Ridge regression (log-linear, log target)
    M3a - Two-stage unit price (global discount curve)
    M3b - Two-stage unit price (per-part discount curves)
    M4  - Kitchen-sink Lasso (log target, interaction terms)
    M5  - Random Forest (raw target)
    M6  - XGBoost (raw target)
    M6b - XGBoost without lead time
    M7  - XGBoost (log target)
    M7b - LightGBM (raw target)
    M7c - LightGBM (log target)
    M8  - Per-estimator XGBoost (3 separate models)
    M9  - Debiased XGBoost (no estimator feature)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from price_estimator.data import compute_unit_price
from price_estimator.features import (
    build_feature_matrix,
    build_feature_matrix_no_estimator,
    extract_description_features,
)

logger = logging.getLogger(__name__)

CV_SEED = 42
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


@dataclass
class CVResults:
    """Results from cross-validation for a single model.

    Attributes:
        model_name: Identifier for the model (e.g., "M0", "M6").
        fold_metrics: List of dicts, one per fold, each with MAPE, MedAPE,
            RMSE, R2 keys.
        mean_metrics: Dict of mean metrics across folds.
        std_metrics: Dict of std metrics across folds.
        all_predictions: Array of predictions (from all folds, in order).
        all_actuals: Array of actual values (from all folds, in order).
        fold_indices: List of arrays, one per fold, containing test indices.
    """

    model_name: str
    fold_metrics: list[dict] = field(default_factory=list)
    mean_metrics: dict = field(default_factory=dict)
    std_metrics: dict = field(default_factory=dict)
    all_predictions: np.ndarray | None = None
    all_actuals: np.ndarray | None = None
    fold_indices: list[np.ndarray] = field(default_factory=list)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAPE, Median APE, RMSE, and R-squared.

    Args:
        y_true: Actual values (must be positive for MAPE).
        y_pred: Predicted values.

    Returns:
        Dict with keys: MAPE, MedAPE, RMSE, R2.
    """
    ape = np.abs(y_true - y_pred) / np.abs(y_true)
    mape = float(np.mean(ape) * 100)
    med_ape = float(np.median(ape) * 100)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"MAPE": mape, "MedAPE": med_ape, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# Base model class
# ---------------------------------------------------------------------------


class BaseModel(ABC):
    """Abstract base class for all price estimator models.

    Subclasses must implement fit() and predict(). The cross_validate()
    method is provided for free and uses the same seeded KFold splits
    across all models.
    """

    name: str = "BaseModel"

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseModel":
        """Train the model on the given DataFrame.

        Args:
            df: Cleaned DataFrame from data.load_data().

        Returns:
            self, for method chaining.
        """
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate price predictions for the given DataFrame.

        Args:
            df: DataFrame with the same columns as training data.

        Returns:
            Array of predicted TotalPrice_USD values.
        """
        ...

    def cross_validate(self, df: pd.DataFrame) -> CVResults:
        """Run K-fold cross-validation with seeded splits.

        Uses the same KFold split (seed=42, n_splits=5) across all models
        to ensure fair comparison.

        Note: Plain KFold (not stratified) is used intentionally. Fold 2
        contains Q-1223 ($115K, dataset max, z=8.6, missing Material and
        Process), which inflates that fold's RMSE/R² but not MAPE. This
        validates MAPE as the primary metric — it's robust to high-value
        outliers that dominate squared-error metrics.

        Args:
            df: Full dataset to cross-validate on.

        Returns:
            CVResults with per-fold and aggregate metrics.
        """
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
        results = CVResults(model_name=self.name)

        all_preds = np.zeros(len(df))
        all_actuals = np.zeros(len(df))

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            self.fit(train_df)
            preds = self.predict(test_df)

            actuals = test_df["TotalPrice_USD"].values
            metrics = compute_metrics(actuals, preds)
            results.fold_metrics.append(metrics)
            results.fold_indices.append(test_idx)

            all_preds[test_idx] = preds
            all_actuals[test_idx] = actuals

            logger.info(
                "%s fold %d: MAPE=%.2f%%, RMSE=%.2f",
                self.name,
                fold_idx,
                metrics["MAPE"],
                metrics["RMSE"],
            )

        results.all_predictions = all_preds
        results.all_actuals = all_actuals

        # Aggregate metrics
        metric_keys = results.fold_metrics[0].keys()
        results.mean_metrics = {
            k: float(np.mean([f[k] for f in results.fold_metrics])) for k in metric_keys
        }
        results.std_metrics = {
            k: float(np.std([f[k] for f in results.fold_metrics])) for k in metric_keys
        }

        return results


# ---------------------------------------------------------------------------
# M0: Lookup Table Baseline
# ---------------------------------------------------------------------------


class M0LookupTable(BaseModel):
    """Pure deterministic formula with no ML.

    base_unit_price = median_unit_price_for_part_type
    total = base_unit_price * material_factor * process_factor
            * qty^discount_slope * (rush_multiplier if rush else 1.0)

    Lookup tables are computed from training data medians.
    """

    name = "M0"

    def __init__(self):
        self.part_type_medians: dict[str, float] = {}
        self.material_factors: dict[str, float] = {}
        self.process_factors: dict[str, float] = {}
        self.rush_multiplier: float = 1.55
        self.discount_slope: float = -0.19
        self.global_median: float = 0.0

    def fit(self, df: pd.DataFrame) -> "M0LookupTable":
        """Compute lookup tables from training data."""
        unit_price = compute_unit_price(df)
        desc_features = extract_description_features(df)

        # Part type median unit prices
        self.global_median = float(unit_price.median())
        for pt in desc_features["base_part_type"].unique():
            mask = desc_features["base_part_type"] == pt
            self.part_type_medians[pt] = float(unit_price[mask].median())

        # Material factors relative to global median
        for mat in df["Material"].dropna().unique():
            mask = df["Material"] == mat
            self.material_factors[mat] = float(unit_price[mask].median() / self.global_median)

        # Process factors relative to global median
        for proc in df["Process"].dropna().unique():
            mask = df["Process"] == proc
            self.process_factors[proc] = float(unit_price[mask].median() / self.global_median)

        # Fit discount slope from data
        log_qty = np.log(df["Quantity"].astype(float))
        log_up = np.log(unit_price)
        from scipy.stats import linregress

        slope, *_ = linregress(log_qty, log_up)
        self.discount_slope = float(slope)

        # Fit rush multiplier from data
        rush_mask = df["RushJob"]
        if rush_mask.sum() > 0 and (~rush_mask).sum() > 0:
            rush_median = float(unit_price[rush_mask].median())
            no_rush_median = float(unit_price[~rush_mask].median())
            self.rush_multiplier = rush_median / no_rush_median

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict prices using lookup tables."""
        desc_features = extract_description_features(df)
        preds = np.zeros(len(df))

        for i in range(len(df)):
            pt = desc_features.iloc[i]["base_part_type"]
            base = self.part_type_medians.get(pt, self.global_median)

            mat = df.iloc[i]["Material"]
            mat_factor = self.material_factors.get(mat, 1.0) if pd.notna(mat) else 1.0

            proc = df.iloc[i]["Process"]
            proc_factor = self.process_factors.get(proc, 1.0) if pd.notna(proc) else 1.0

            qty = df.iloc[i]["Quantity"]
            # Unit price * quantity with discount curve
            # base already is a unit price; apply factors, then scale by quantity
            unit_price = base * mat_factor * proc_factor
            rush = self.rush_multiplier if df.iloc[i]["RushJob"] else 1.0

            # qty^(1 + slope) gives total price with discount
            # since unit_price * qty^1 = total at no discount,
            # and unit_price * qty^(1+slope) bakes in the discount
            total = unit_price * (qty ** (1 + self.discount_slope)) * rush
            preds[i] = total

        return preds


# ---------------------------------------------------------------------------
# M1: Ridge (additive, raw target)
# ---------------------------------------------------------------------------


class M1RidgeAdditive(BaseModel):
    """Ridge regression on raw price with one-hot encoded features.

    Uses StandardScaler to avoid numerical overflow in RidgeCV.
    """

    name = "M1"

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 20))),
            ]
        )
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> "M1RidgeAdditive":
        X, y = build_feature_matrix(df, encoding="onehot")
        X = X.fillna(0)
        self._feature_cols = list(X.columns)
        self.pipeline.fit(X.values, y.values)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = build_feature_matrix(df, encoding="onehot")
        X = X.fillna(0)
        X = X.reindex(columns=self._feature_cols, fill_value=0)
        preds = self.pipeline.predict(X.values)
        return np.maximum(preds, 0)


# ---------------------------------------------------------------------------
# M2: Ridge (log-linear, log target)
# ---------------------------------------------------------------------------


class M2RidgeLogLinear(BaseModel):
    """Ridge regression on log(price) with one-hot features.

    Uses StandardScaler to avoid numerical overflow. Applies Jensen's
    inequality correction when back-transforming:
    corrected = exp(log_pred + 0.5 * sigma^2)
    """

    name = "M2"

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 20))),
            ]
        )
        self._feature_cols: list[str] = []
        self._residual_var: float = 0.0

    def fit(self, df: pd.DataFrame) -> "M2RidgeLogLinear":
        X, y = build_feature_matrix(df, encoding="onehot")
        X = X.fillna(0)
        self._feature_cols = list(X.columns)
        log_y = np.log(y.values)
        self.pipeline.fit(X.values, log_y)

        # Compute residual variance for Jensen's correction
        log_preds = self.pipeline.predict(X.values)
        residuals = log_y - log_preds
        self._residual_var = float(np.var(residuals))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = build_feature_matrix(df, encoding="onehot")
        X = X.fillna(0)
        X = X.reindex(columns=self._feature_cols, fill_value=0)
        log_preds = self.pipeline.predict(X.values)
        # Jensen's correction: exp(mu + 0.5 * sigma^2)
        return np.exp(log_preds + 0.5 * self._residual_var)


# ---------------------------------------------------------------------------
# M3a/M3b: Two-stage unit price models
# ---------------------------------------------------------------------------


class M3TwoStage(BaseModel):
    """Two-stage model: predict unit price, apply quantity discount curve.

    Stage 1: Ridge on log(unit_price) from part/material/process/estimator
    Stage 2: Quantity discount as qty^(1+slope)
    Rush multiplier applied on top.

    Args:
        per_part_curves: If True (M3b), fit discount curves per part type.
            If False (M3a), fit a single global curve.
    """

    def __init__(self, per_part_curves: bool = False):
        self.per_part_curves = per_part_curves
        self.name = "M3b" if per_part_curves else "M3a"
        self.unit_price_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 20))),
            ]
        )
        self._feature_cols: list[str] = []
        self._residual_var: float = 0.0
        self.global_slope: float = 0.0
        self.part_slopes: dict[str, float] = {}
        self.rush_multiplier: float = 1.0

    def fit(self, df: pd.DataFrame) -> "M3TwoStage":
        from scipy.stats import linregress

        unit_price = compute_unit_price(df)
        log_up = np.log(unit_price)
        log_qty = np.log(df["Quantity"].astype(float))

        # Stage 2 first: fit quantity discount curves
        slope, *_ = linregress(log_qty.values, log_up.values)
        self.global_slope = float(slope)

        if self.per_part_curves:
            desc_features = extract_description_features(df)
            for pt in desc_features["base_part_type"].unique():
                mask = desc_features["base_part_type"] == pt
                if mask.sum() >= 5:
                    pt_slope, *_ = linregress(log_qty.values[mask], log_up.values[mask])
                    self.part_slopes[pt] = float(pt_slope)
                else:
                    self.part_slopes[pt] = self.global_slope

        # Rush multiplier
        rush_mask = df["RushJob"]
        if rush_mask.sum() > 0 and (~rush_mask).sum() > 0:
            self.rush_multiplier = float(
                unit_price[rush_mask].median() / unit_price[~rush_mask].median()
            )

        # Stage 1: predict qty=1 equivalent unit price
        # Remove quantity effect and rush effect from unit price
        desc_features = extract_description_features(df)
        if self.per_part_curves:
            qty_adjustment = np.array(
                [
                    self.part_slopes.get(pt, self.global_slope)
                    for pt in desc_features["base_part_type"]
                ]
            )
        else:
            qty_adjustment = np.full(len(df), self.global_slope)

        # Normalize to qty=1 equivalent
        log_up_normalized = log_up.values - qty_adjustment * log_qty.values
        rush_adj = np.where(df["RushJob"].values, np.log(self.rush_multiplier), 0)
        log_up_normalized = log_up_normalized - rush_adj

        # Build feature matrix (without log_qty since we handle it separately)
        X, _ = build_feature_matrix(df, encoding="onehot")
        # Drop quantity-related columns
        drop_cols = [
            c
            for c in X.columns
            if "quantity" in c.lower() or "rush" in c.lower() or "lead_time" in c.lower()
        ]
        X = X.drop(columns=drop_cols, errors="ignore")
        X = X.fillna(0)
        self._feature_cols = list(X.columns)

        self.unit_price_pipeline.fit(X.values, log_up_normalized)
        residuals = log_up_normalized - self.unit_price_pipeline.predict(X.values)
        self._residual_var = float(np.var(residuals))

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = build_feature_matrix(df, encoding="onehot")
        drop_cols = [
            c
            for c in X.columns
            if "quantity" in c.lower() or "rush" in c.lower() or "lead_time" in c.lower()
        ]
        X = X.drop(columns=drop_cols, errors="ignore")
        X = X.fillna(0)
        X = X.reindex(columns=self._feature_cols, fill_value=0)

        log_up_pred = self.unit_price_pipeline.predict(X.values)

        # Apply Jensen's correction
        log_up_pred = log_up_pred + 0.5 * self._residual_var

        # Apply quantity discount and rush
        log_qty = np.log(df["Quantity"].astype(float).values)

        if self.per_part_curves:
            desc_features = extract_description_features(df)
            slopes = np.array(
                [
                    self.part_slopes.get(pt, self.global_slope)
                    for pt in desc_features["base_part_type"]
                ]
            )
        else:
            slopes = np.full(len(df), self.global_slope)

        # total = unit_price_at_qty1 * qty^(1+slope) * rush
        log_total = log_up_pred + (1 + slopes) * log_qty
        rush_adj = np.where(df["RushJob"].values, np.log(self.rush_multiplier), 0)
        log_total = log_total + rush_adj

        return np.exp(log_total)


# ---------------------------------------------------------------------------
# M4: Kitchen-sink Lasso
# ---------------------------------------------------------------------------


class M4KitchenSinkLasso(BaseModel):
    """Lasso on log(price) with all features including interaction terms.

    Includes Material x Process, Material x PartType, Rush x Material,
    Rush x PartType interactions. Regularization selects relevant features.
    """

    name = "M4"

    def __init__(self):
        # Inner cv=5 selects alpha via nested CV on the training fold.
        # This is properly nested — no leakage with the outer 5-fold CV.
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lasso", LassoCV(alphas=np.logspace(-4, 2, 30), cv=5, max_iter=10000)),
            ]
        )
        self._feature_cols: list[str] = []
        self._residual_var: float = 0.0

    def _build_kitchen_sink_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix with interaction terms."""
        X, _ = build_feature_matrix(df, encoding="onehot")
        X = X.fillna(0)

        # Add date features
        X["days_since_epoch"] = (df["Date"] - pd.Timestamp("2023-01-01")).dt.days

        # Add interaction terms for rush × material and rush × part type
        rush = X["rush_job"].values
        material_cols = [c for c in X.columns if c.startswith("material_")]
        part_cols = [c for c in X.columns if c.startswith("base_part_type_")]

        for col in material_cols:
            X[f"rush_x_{col}"] = rush * X[col].values
        for col in part_cols:
            X[f"rush_x_{col}"] = rush * X[col].values

        # Material cost tier × log quantity
        X["tier_x_log_qty"] = X["material_cost_tier"] * X["log_quantity"]

        # Rush × lead time
        X["rush_x_lead_time"] = rush * X["lead_time_weeks"].values

        return X

    def fit(self, df: pd.DataFrame) -> "M4KitchenSinkLasso":
        X = self._build_kitchen_sink_features(df)
        self._feature_cols = list(X.columns)
        log_y = np.log(df["TotalPrice_USD"].values)
        self.pipeline.fit(X.values, log_y)

        log_preds = self.pipeline.predict(X.values)
        residuals = log_y - log_preds
        self._residual_var = float(np.var(residuals))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._build_kitchen_sink_features(df)
        X = X.reindex(columns=self._feature_cols, fill_value=0)
        log_preds = self.pipeline.predict(X.values)
        return np.exp(log_preds + 0.5 * self._residual_var)


# ---------------------------------------------------------------------------
# M5: Random Forest
# ---------------------------------------------------------------------------


class M5RandomForest(BaseModel):
    """Random Forest regressor with label-encoded features."""

    name = "M5"

    def __init__(self, **kwargs):
        defaults = {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 5,
            "random_state": CV_SEED,
            "n_jobs": -1,
        }
        defaults.update(kwargs)
        self.model = RandomForestRegressor(**defaults)
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> "M5RandomForest":
        X, y = build_feature_matrix(df, encoding="label")
        X = X.fillna(-1)  # Label encoding: -1 for missing
        self._feature_cols = list(X.columns)
        self.model.fit(X.values, y.values)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = build_feature_matrix(df, encoding="label")
        X = X.fillna(-1)
        X = X.reindex(columns=self._feature_cols, fill_value=-1)
        return self.model.predict(X.values)


# ---------------------------------------------------------------------------
# M6/M6b/M7: XGBoost variants
# ---------------------------------------------------------------------------


class XGBoostModel(BaseModel):
    """XGBoost regressor with configurable target transform and features.

    Args:
        name: Model identifier.
        log_target: If True, train on log(price) with Jensen's correction.
        exclude_lead_time: If True, drop lead_time_weeks from features.
        exclude_estimator: If True, drop estimator columns.
        xgb_params: Additional XGBoost parameters.
    """

    def __init__(
        self,
        name: str = "M6",
        log_target: bool = False,
        exclude_lead_time: bool = False,
        exclude_estimator: bool = False,
        xgb_params: dict | None = None,
    ):
        self.name = name
        self.log_target = log_target
        self.exclude_lead_time = exclude_lead_time
        self.exclude_estimator = exclude_estimator
        self._feature_cols: list[str] = []
        self._residual_var: float = 0.0

        import xgboost as xgb

        defaults = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "random_state": CV_SEED,
            "n_jobs": -1,
        }
        if xgb_params:
            defaults.update(xgb_params)
        self.model = xgb.XGBRegressor(**defaults)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.exclude_estimator:
            X, _ = build_feature_matrix_no_estimator(df, encoding="label")
        else:
            X, _ = build_feature_matrix(df, encoding="label")
        X = X.fillna(-1)
        if self.exclude_lead_time:
            X = X.drop(columns=["lead_time_weeks"], errors="ignore")
        return X

    def fit(self, df: pd.DataFrame) -> "XGBoostModel":
        X = self._prepare_features(df)
        self._feature_cols = list(X.columns)
        y = df["TotalPrice_USD"].values
        if self.log_target:
            y = np.log(y)

        self.model.fit(X.values, y)

        if self.log_target:
            preds = self.model.predict(X.values)
            residuals = y - preds
            self._residual_var = float(np.var(residuals))

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_features(df)
        X = X.reindex(columns=self._feature_cols, fill_value=-1)
        preds = self.model.predict(X.values)
        if self.log_target:
            return np.exp(preds + 0.5 * self._residual_var)
        return np.maximum(preds, 0)


# ---------------------------------------------------------------------------
# M7b/M7c: LightGBM variants
# ---------------------------------------------------------------------------


class LightGBMModel(BaseModel):
    """LightGBM regressor with configurable target transform.

    Args:
        name: Model identifier.
        log_target: If True, train on log(price) with Jensen's correction.
        lgbm_params: Additional LightGBM parameters.
    """

    def __init__(
        self,
        name: str = "M7b",
        log_target: bool = False,
        lgbm_params: dict | None = None,
    ):
        self.name = name
        self.log_target = log_target
        self._feature_cols: list[str] = []
        self._residual_var: float = 0.0

        import lightgbm as lgb

        defaults = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 10,
            "random_state": CV_SEED,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if lgbm_params:
            defaults.update(lgbm_params)
        self.model = lgb.LGBMRegressor(**defaults)

    def fit(self, df: pd.DataFrame) -> "LightGBMModel":
        X, _ = build_feature_matrix(df, encoding="label")
        X = X.fillna(-1)
        self._feature_cols = list(X.columns)
        y = df["TotalPrice_USD"].values
        if self.log_target:
            y = np.log(y)

        self.model.fit(X.values, y)

        if self.log_target:
            preds = self.model.predict(X.values)
            residuals = y - preds
            self._residual_var = float(np.var(residuals))

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = build_feature_matrix(df, encoding="label")
        X = X.fillna(-1)
        X = X.reindex(columns=self._feature_cols, fill_value=-1)
        preds = self.model.predict(X.values)
        if self.log_target:
            return np.exp(preds + 0.5 * self._residual_var)
        return np.maximum(preds, 0)


# ---------------------------------------------------------------------------
# M8: Per-estimator XGBoost
# ---------------------------------------------------------------------------


class M8PerEstimatorXGBoost(BaseModel):
    """Three separate XGBoost models, one per estimator.

    Tests whether estimator bias is structural (per-estimator wins)
    or additive (single model with estimator feature wins).

    Note: Each per-estimator model trains on ~136 rows during CV
    (510 / 3 estimators * 4/5 folds), which is marginal for XGBoost.
    Poor M8 performance (55.6% MAPE) is expected and confirms the
    additive bias hypothesis — estimator differences are better
    captured as a feature in a single model (M6) than as separate
    models with reduced training data.
    """

    name = "M8"

    def __init__(self, xgb_params: dict | None = None):
        self.models: dict[str, XGBoostModel] = {}
        self.xgb_params = xgb_params or {}

    def fit(self, df: pd.DataFrame) -> "M8PerEstimatorXGBoost":
        for estimator in df["Estimator"].unique():
            mask = df["Estimator"] == estimator
            subset = df[mask].copy()
            model = XGBoostModel(
                name=f"M8_{estimator}",
                exclude_estimator=True,
                xgb_params=self.xgb_params,
            )
            model.fit(subset)
            self.models[estimator] = model
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df))
        for estimator, model in self.models.items():
            mask = df["Estimator"] == estimator
            if mask.sum() > 0:
                preds[mask.values] = model.predict(df[mask])

        # Guard: fallback for estimators absent from training
        missing_estimators = set(df["Estimator"].unique()) - set(self.models.keys())
        if missing_estimators:
            logger.warning(
                "M8: estimators %s absent from training, using ensemble fallback",
                missing_estimators,
            )
            for est in missing_estimators:
                mask = df["Estimator"] == est
                if mask.sum() > 0:
                    sub_preds = np.column_stack(
                        [m.predict(df[mask]) for m in self.models.values()]
                    )
                    preds[mask.values] = np.mean(sub_preds, axis=1)

        return preds


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_all_models() -> list[BaseModel]:
    """Create instances of all models in the matrix.

    Returns:
        List of model instances ready for cross-validation.
    """
    return [
        M0LookupTable(),
        M1RidgeAdditive(),
        M2RidgeLogLinear(),
        M3TwoStage(per_part_curves=False),  # M3a
        M3TwoStage(per_part_curves=True),  # M3b
        M4KitchenSinkLasso(),
        M5RandomForest(),
        XGBoostModel(name="M6"),
        XGBoostModel(name="M6b", exclude_lead_time=True),
        XGBoostModel(name="M7", log_target=True),
        LightGBMModel(name="M7b"),
        LightGBMModel(name="M7c", log_target=True),
        M8PerEstimatorXGBoost(),
        XGBoostModel(name="M9", exclude_estimator=True),
    ]


def get_model_by_name(name: str) -> BaseModel:
    """Get a single model instance by name.

    Args:
        name: Model name (e.g., "M0", "M6", "M9").

    Returns:
        Model instance.

    Raises:
        ValueError: If name is not recognized.
    """
    for model in get_all_models():
        if model.name == name:
            return model
    valid = [m.name for m in get_all_models()]
    raise ValueError(f"Unknown model '{name}'. Valid names: {valid}")


def run_all_cv(df: pd.DataFrame, model_names: list[str] | None = None) -> list[CVResults]:
    """Run cross-validation for all models (or a subset).

    Args:
        df: Full dataset.
        model_names: Optional list of model names to run. If None, runs all.

    Returns:
        List of CVResults, one per model.
    """
    models = get_all_models()
    if model_names:
        models = [m for m in models if m.name in model_names]

    results = []
    for model in models:
        logger.info("Cross-validating %s...", model.name)
        try:
            cv_result = model.cross_validate(df)
            results.append(cv_result)
            logger.info(
                "%s: MAPE=%.2f%% (+/- %.2f%%)",
                model.name,
                cv_result.mean_metrics["MAPE"],
                cv_result.std_metrics["MAPE"],
            )
        except Exception:
            logger.exception("Failed to cross-validate %s", model.name)

    return results


def results_to_dataframe(results: list[CVResults]) -> pd.DataFrame:
    """Convert CV results to a comparison DataFrame.

    Args:
        results: List of CVResults from run_all_cv().

    Returns:
        DataFrame with one row per model and columns for each metric
        (mean and std).
    """
    rows = []
    for r in results:
        row = {"model": r.model_name}
        for k, v in r.mean_metrics.items():
            row[f"{k}_mean"] = v
        for k, v in r.std_metrics.items():
            row[f"{k}_std"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("MAPE_mean").reset_index(drop=True)
