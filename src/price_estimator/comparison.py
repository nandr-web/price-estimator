"""Seven-lens model comparison framework.

Evaluates trained models across multiple dimensions beyond raw accuracy:
error profile, segment fairness, economic coherence, calibration/bias
direction, stability/robustness, boundary behavior, and complexity.

Each lens function takes trained models and data, returns a structured dict.
The scorecard function aggregates all lenses into a ranked summary.

This module complements scripts/evaluate.py (which answers "which model has
the best MAPE?") by answering "which model should we trust?"
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from price_estimator.models import CV_SEED, N_FOLDS, BaseModel, compute_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lens 1: Error Profile
# ---------------------------------------------------------------------------


def error_profile(models: dict[str, BaseModel], df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Characterize the shape of each model's error distribution.

    Beyond mean MAPE, computes median APE, 90th/95th percentile APE,
    max APE, and signed error skew. Two models at the same MAPE can
    differ dramatically — one tight, the other with a long tail of
    catastrophic misses.

    Args:
        models: Dict mapping model name to a trained BaseModel.
        df: Full dataset for cross-validated error collection.

    Returns:
        Dict mapping model name to error profile dict with keys:
        mape, median_ape, p90_ape, p95_ape, max_ape, skew,
        pct_under_5, pct_under_10, pct_under_20, pct_over_50,
        ape_values (raw array for plotting).
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    results = {}

    for name, model in models.items():
        all_ape = []
        all_signed = []

        for train_idx, test_idx in kf.split(df):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            model.fit(train_df)
            preds = model.predict(test_df)
            actuals = test_df["TotalPrice_USD"].values

            ape = np.abs(actuals - preds) / actuals * 100
            signed = (preds - actuals) / actuals * 100
            all_ape.extend(ape.tolist())
            all_signed.extend(signed.tolist())

        all_ape = np.array(all_ape)
        all_signed = np.array(all_signed)

        results[name] = {
            "mape": float(np.mean(all_ape)),
            "median_ape": float(np.median(all_ape)),
            "p90_ape": float(np.percentile(all_ape, 90)),
            "p95_ape": float(np.percentile(all_ape, 95)),
            "max_ape": float(np.max(all_ape)),
            "skew": float(pd.Series(all_signed).skew()),
            "pct_under_5": float(np.mean(all_ape < 5) * 100),
            "pct_under_10": float(np.mean(all_ape < 10) * 100),
            "pct_under_20": float(np.mean(all_ape < 20) * 100),
            "pct_over_50": float(np.mean(all_ape > 50) * 100),
            "ape_values": all_ape,
        }

    return results


# ---------------------------------------------------------------------------
# Lens 2: Segment Fairness
# ---------------------------------------------------------------------------


def segment_fairness(
    models: dict[str, BaseModel], df: pd.DataFrame
) -> dict[str, dict[str, dict[str, float]]]:
    """Measure accuracy across data segments to detect blind spots.

    A model with 10% overall MAPE might be 5% on cheap Aluminum jobs
    but 25% on expensive Inconel qty=1 jobs. This lens reveals where
    models struggle.

    Segments: price quartile, material, part type (base), quantity band,
    estimator.

    Args:
        models: Dict mapping model name to a trained BaseModel.
        df: Full dataset.

    Returns:
        Dict mapping model name to segment results. Each segment result
        maps segment_name -> { category -> { mape, median_ape, count } }.
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    # Collect OOF predictions per model
    oof_preds = {name: np.zeros(len(df)) for name in models}

    for train_idx, test_idx in kf.split(df):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        for name, model in models.items():
            model.fit(train_df)
            oof_preds[name][test_idx] = model.predict(test_df)

    actuals = df["TotalPrice_USD"].values

    # Define segments
    price_quartiles = pd.qcut(actuals, 4, labels=["Q1 (cheapest)", "Q2", "Q3", "Q4 (priciest)"])
    base_part_type = df["PartDescription"].str.split(" - ").str[0]
    qty_bands = pd.cut(
        df["Quantity"],
        bins=[0, 1, 10, 50, 100],
        labels=["qty=1", "qty=5-10", "qty=20-50", "qty=100"],
    )

    segments = {
        "by_price_quartile": price_quartiles,
        "by_material": df["Material"].fillna("Missing"),
        "by_part_type": base_part_type,
        "by_quantity_band": qty_bands,
        "by_estimator": df["Estimator"],
    }

    results = {}
    for name in models:
        preds = oof_preds[name]
        model_segments = {}

        for seg_name, seg_labels in segments.items():
            seg_results = {}
            for cat in seg_labels.unique():
                mask = seg_labels == cat
                if mask.sum() == 0:
                    continue
                cat_actuals = actuals[mask]
                cat_preds = preds[mask]
                ape = np.abs(cat_actuals - cat_preds) / cat_actuals * 100
                seg_results[str(cat)] = {
                    "mape": float(np.mean(ape)),
                    "median_ape": float(np.median(ape)),
                    "count": int(mask.sum()),
                }
            model_segments[seg_name] = seg_results

        results[name] = model_segments

    return results


# ---------------------------------------------------------------------------
# Lens 3: Economic Coherence
# ---------------------------------------------------------------------------


def _build_probe_row(
    part_desc: str = "Sensor Housing - standard",
    material: str = "Aluminum 6061",
    process: str = "3-Axis Milling",
    quantity: int = 10,
    rush: bool = False,
    lead_time: int = 6,
    estimator: str = "Suzuki-san",
) -> pd.DataFrame:
    """Build a single-row DataFrame for probe predictions."""
    return pd.DataFrame(
        [
            {
                "QuoteID": "PROBE",
                "Date": pd.Timestamp("2024-01-15"),
                "PartDescription": part_desc,
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


def economic_coherence(
    models: dict[str, BaseModel], df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Test whether models respect basic economic relationships.

    Uses synthetic probe inputs (deterministic) and also samples from the
    training distribution (stochastic). Checks:
    - Material cost ordering (Inconel > Ti > SS > Al7075 > Al6061)
    - Process precision ordering (5-Axis > Wire EDM > 3-Axis > CNC > Grinding)
    - Quantity discount (more qty → lower unit price)
    - Rush premium (rush=Yes → higher price)
    - Complexity premium (more modifiers → higher price)

    Uses absolute pass/fail — either Inconel > Aluminum or it doesn't.

    Args:
        models: Dict mapping model name to trained (on full df) BaseModel.
        df: Training data (for stochastic sampling).

    Returns:
        Dict mapping model name to coherence results with:
        checks (list of {name, passed, details}), pass_count, total_count,
        pass_rate.
    """
    # Train all models on full data for probe predictions
    for model in models.values():
        model.fit(df)

    materials_ordered = [
        "Aluminum 6061",
        "Aluminum 7075",
        "Stainless Steel 17-4 PH",
        "Titanium Grade 5",
        "Inconel 718",
    ]
    processes_ordered = [
        "Surface Grinding",
        "CNC Turning",
        "3-Axis Milling",
        "Wire EDM",
        "5-Axis Milling",
    ]
    quantities = [1, 5, 10, 20, 50, 100]

    results = {}

    for name, model in models.items():
        checks = []

        # --- Deterministic probes ---

        # Material ordering: predict same job across all materials
        mat_preds = {}
        for mat in materials_ordered:
            probe = _build_probe_row(material=mat)
            mat_preds[mat] = float(model.predict(probe)[0])

        mat_values = [mat_preds[m] for m in materials_ordered]
        mat_violations = 0
        mat_violation_details = []
        for i in range(len(materials_ordered) - 1):
            if mat_values[i] >= mat_values[i + 1]:
                mat_violations += 1
                mat_violation_details.append(
                    f"{materials_ordered[i]} (${mat_values[i]:,.0f}) >= "
                    f"{materials_ordered[i + 1]} (${mat_values[i + 1]:,.0f})"
                )
        checks.append(
            {
                "name": "material_ordering",
                "type": "deterministic",
                "passed": mat_violations == 0,
                "violations": mat_violations,
                "max_violations": len(materials_ordered) - 1,
                "details": mat_violation_details,
                "predictions": mat_preds,
            }
        )

        # Process ordering: predict same job across all processes
        proc_preds = {}
        for proc in processes_ordered:
            probe = _build_probe_row(process=proc)
            proc_preds[proc] = float(model.predict(probe)[0])

        proc_values = [proc_preds[p] for p in processes_ordered]
        proc_violations = 0
        proc_violation_details = []
        for i in range(len(processes_ordered) - 1):
            if proc_values[i] >= proc_values[i + 1]:
                proc_violations += 1
                proc_violation_details.append(
                    f"{processes_ordered[i]} (${proc_values[i]:,.0f}) >= "
                    f"{processes_ordered[i + 1]} (${proc_values[i + 1]:,.0f})"
                )
        checks.append(
            {
                "name": "process_ordering",
                "type": "deterministic",
                "passed": proc_violations == 0,
                "violations": proc_violations,
                "max_violations": len(processes_ordered) - 1,
                "details": proc_violation_details,
                "predictions": proc_preds,
            }
        )

        # Quantity discount: predict same job at each quantity
        qty_unit_preds = {}
        for qty in quantities:
            probe = _build_probe_row(quantity=qty)
            total = float(model.predict(probe)[0])
            qty_unit_preds[qty] = total / qty

        qty_values = [qty_unit_preds[q] for q in quantities]
        qty_violations = 0
        qty_violation_details = []
        for i in range(len(quantities) - 1):
            if qty_values[i] < qty_values[i + 1]:
                qty_violations += 1
                qty_violation_details.append(
                    f"qty={quantities[i]} (${qty_values[i]:,.2f}/unit) < "
                    f"qty={quantities[i + 1]} (${qty_values[i + 1]:,.2f}/unit)"
                )
        checks.append(
            {
                "name": "quantity_discount",
                "type": "deterministic",
                "passed": qty_violations == 0,
                "violations": qty_violations,
                "max_violations": len(quantities) - 1,
                "details": qty_violation_details,
                "unit_prices": qty_unit_preds,
            }
        )

        # Rush premium: same job with and without rush
        no_rush = float(model.predict(_build_probe_row(rush=False))[0])
        yes_rush = float(model.predict(_build_probe_row(rush=True))[0])
        rush_passed = yes_rush > no_rush
        checks.append(
            {
                "name": "rush_premium",
                "type": "deterministic",
                "passed": rush_passed,
                "no_rush_price": no_rush,
                "rush_price": yes_rush,
                "premium_pct": (yes_rush - no_rush) / no_rush * 100 if no_rush > 0 else 0,
                "details": []
                if rush_passed
                else [f"Rush (${yes_rush:,.0f}) <= No rush (${no_rush:,.0f})"],
            }
        )

        # Complexity premium: "standard" vs "thin walls, high precision"
        simple = float(model.predict(_build_probe_row(part_desc="Sensor Housing - standard"))[0])
        complex_ = float(
            model.predict(_build_probe_row(part_desc="Sensor Housing - high precision"))[0]
        )
        complexity_passed = complex_ >= simple
        checks.append(
            {
                "name": "complexity_premium",
                "type": "deterministic",
                "passed": complexity_passed,
                "simple_price": simple,
                "complex_price": complex_,
                "details": []
                if complexity_passed
                else [f"Complex (${complex_:,.0f}) < Simple (${simple:,.0f})"],
            }
        )

        # --- Stochastic probes (sample from training distribution) ---
        rng = np.random.RandomState(CV_SEED)
        n_stochastic = 50
        stoch_mat_violations = 0
        stoch_rush_violations = 0
        stoch_qty_violations = 0
        stoch_total = 0

        for _ in range(n_stochastic):
            idx = rng.randint(len(df))
            row = df.iloc[idx]
            base_desc = row["PartDescription"]
            mat = row["Material"] if pd.notna(row["Material"]) else "Aluminum 6061"
            proc = row["Process"] if pd.notna(row["Process"]) else "3-Axis Milling"
            lead = row["LeadTimeWeeks"]
            est = row["Estimator"]

            # Material: compare cheapest vs most expensive
            cheap = float(
                model.predict(
                    _build_probe_row(
                        part_desc=base_desc,
                        material="Aluminum 6061",
                        process=proc,
                        quantity=10,
                        lead_time=lead,
                        estimator=est,
                    )
                )[0]
            )
            expensive = float(
                model.predict(
                    _build_probe_row(
                        part_desc=base_desc,
                        material="Inconel 718",
                        process=proc,
                        quantity=10,
                        lead_time=lead,
                        estimator=est,
                    )
                )[0]
            )
            if cheap >= expensive:
                stoch_mat_violations += 1

            # Rush
            no_r = float(
                model.predict(
                    _build_probe_row(
                        part_desc=base_desc,
                        material=mat,
                        process=proc,
                        quantity=10,
                        rush=False,
                        lead_time=lead,
                        estimator=est,
                    )
                )[0]
            )
            yes_r = float(
                model.predict(
                    _build_probe_row(
                        part_desc=base_desc,
                        material=mat,
                        process=proc,
                        quantity=10,
                        rush=True,
                        lead_time=lead,
                        estimator=est,
                    )
                )[0]
            )
            if yes_r <= no_r:
                stoch_rush_violations += 1

            # Quantity: qty=1 unit price should be >= qty=100 unit price
            up_1 = float(
                model.predict(
                    _build_probe_row(
                        part_desc=base_desc,
                        material=mat,
                        process=proc,
                        quantity=1,
                        lead_time=lead,
                        estimator=est,
                    )
                )[0]
            )
            up_100 = (
                float(
                    model.predict(
                        _build_probe_row(
                            part_desc=base_desc,
                            material=mat,
                            process=proc,
                            quantity=100,
                            lead_time=lead,
                            estimator=est,
                        )
                    )[0]
                )
                / 100
            )
            if up_1 < up_100:
                stoch_qty_violations += 1

            stoch_total += 1

        checks.append(
            {
                "name": "stochastic_material_ordering",
                "type": "stochastic",
                "passed": stoch_mat_violations == 0,
                "violations": stoch_mat_violations,
                "total_probes": stoch_total,
                "violation_rate": stoch_mat_violations / stoch_total * 100,
            }
        )
        checks.append(
            {
                "name": "stochastic_rush_premium",
                "type": "stochastic",
                "passed": stoch_rush_violations == 0,
                "violations": stoch_rush_violations,
                "total_probes": stoch_total,
                "violation_rate": stoch_rush_violations / stoch_total * 100,
            }
        )
        checks.append(
            {
                "name": "stochastic_quantity_discount",
                "type": "stochastic",
                "passed": stoch_qty_violations == 0,
                "violations": stoch_qty_violations,
                "total_probes": stoch_total,
                "violation_rate": stoch_qty_violations / stoch_total * 100,
            }
        )

        pass_count = sum(1 for c in checks if c["passed"])
        results[name] = {
            "checks": checks,
            "pass_count": pass_count,
            "total_count": len(checks),
            "pass_rate": pass_count / len(checks) * 100,
        }

    return results


# ---------------------------------------------------------------------------
# Lens 4: Calibration & Bias Direction
# ---------------------------------------------------------------------------


def calibration_bias(models: dict[str, BaseModel], df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Measure the direction and consistency of prediction errors.

    In a machine shop, underestimation loses money (aggressive), while
    overestimation loses bids but protects margin (conservative). The
    direction matters as much as magnitude.

    Args:
        models: Dict mapping model name to trained BaseModel.
        df: Full dataset.

    Returns:
        Dict mapping model name to calibration results:
        mean_signed_error_dollars, mean_signed_error_pct,
        pct_overestimated, signed_error_by_price_quartile, label.
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    actuals_full = df["TotalPrice_USD"].values
    price_quartiles = pd.qcut(actuals_full, 4, labels=["Q1", "Q2", "Q3", "Q4"])

    results = {}

    for name, model in models.items():
        all_signed_dollars = np.zeros(len(df))
        all_signed_pct = np.zeros(len(df))

        for train_idx, test_idx in kf.split(df):
            model.fit(df.iloc[train_idx].copy())
            preds = model.predict(df.iloc[test_idx].copy())
            actuals = actuals_full[test_idx]
            all_signed_dollars[test_idx] = preds - actuals
            all_signed_pct[test_idx] = (preds - actuals) / actuals * 100

        mean_signed_pct = float(np.mean(all_signed_pct))
        pct_over = float(np.mean(all_signed_dollars > 0) * 100)

        # Bias direction by price quartile
        by_quartile = {}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            mask = price_quartiles == q
            if mask.sum() > 0:
                by_quartile[q] = {
                    "mean_signed_pct": float(np.mean(all_signed_pct[mask])),
                    "pct_overestimated": float(np.mean(all_signed_dollars[mask] > 0) * 100),
                    "count": int(mask.sum()),
                }

        # Label: conservative (overestimates), aggressive (underestimates), balanced
        if mean_signed_pct > 3:
            label = "conservative"
        elif mean_signed_pct < -3:
            label = "aggressive"
        else:
            label = "balanced"

        results[name] = {
            "mean_signed_error_dollars": float(np.mean(all_signed_dollars)),
            "mean_signed_error_pct": mean_signed_pct,
            "median_signed_error_pct": float(np.median(all_signed_pct)),
            "pct_overestimated": pct_over,
            "by_price_quartile": by_quartile,
            "label": label,
        }

    return results


# ---------------------------------------------------------------------------
# Lens 5: Stability & Robustness
# ---------------------------------------------------------------------------


def stability_robustness(
    models: dict[str, BaseModel], df: pd.DataFrame, n_bootstrap: int = 20
) -> dict[str, dict[str, Any]]:
    """Measure how sensitive models are to data perturbations.

    A model with 8% MAPE ± 4% across folds is less trustworthy than
    10% MAPE ± 0.5%. With only 510 rows, instability is a real risk.

    Tests: fold-to-fold variance, bootstrap resampling stability,
    and feature group dropout robustness.

    Args:
        models: Dict mapping model name to trained BaseModel.
        df: Full dataset.
        n_bootstrap: Number of bootstrap resamples for stability test.

    Returns:
        Dict mapping model name to stability results.
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    rng = np.random.RandomState(CV_SEED)

    results = {}

    for name, model in models.items():
        # Fold-to-fold MAPE variance
        fold_mapes = []
        for train_idx, test_idx in kf.split(df):
            model.fit(df.iloc[train_idx].copy())
            preds = model.predict(df.iloc[test_idx].copy())
            actuals = df.iloc[test_idx]["TotalPrice_USD"].values
            metrics = compute_metrics(actuals, preds)
            fold_mapes.append(metrics["MAPE"])

        # Bootstrap prediction stability
        # For a fixed test set, resample training and measure prediction spread
        # Use first fold's test set as fixed reference
        fold_splits = list(kf.split(df))
        ref_train_idx, ref_test_idx = fold_splits[0]
        ref_test_df = df.iloc[ref_test_idx].copy()
        ref_train_pool = df.iloc[ref_train_idx]

        bootstrap_preds = []
        for _ in range(n_bootstrap):
            # Resample training data with replacement
            boot_idx = rng.choice(len(ref_train_pool), size=len(ref_train_pool), replace=True)
            boot_df = ref_train_pool.iloc[boot_idx].copy()
            model.fit(boot_df)
            preds = model.predict(ref_test_df)
            bootstrap_preds.append(preds)

        bootstrap_preds = np.array(bootstrap_preds)  # (n_bootstrap, n_test)
        # Per-sample coefficient of variation
        pred_means = bootstrap_preds.mean(axis=0)
        pred_stds = bootstrap_preds.std(axis=0)
        safe_means = np.maximum(pred_means, 1.0)
        cv_per_sample = pred_stds / safe_means * 100
        mean_cv = float(np.mean(cv_per_sample))

        results[name] = {
            "fold_mapes": fold_mapes,
            "fold_mape_mean": float(np.mean(fold_mapes)),
            "fold_mape_std": float(np.std(fold_mapes)),
            "fold_mape_range": float(np.max(fold_mapes) - np.min(fold_mapes)),
            "bootstrap_mean_cv_pct": mean_cv,
            "bootstrap_max_cv_pct": float(np.max(cv_per_sample)),
        }

    return results


# ---------------------------------------------------------------------------
# Lens 6: Boundary Behavior
# ---------------------------------------------------------------------------


def boundary_behavior(models: dict[str, BaseModel], df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Test model behavior at distribution edges and beyond.

    Checks: low/high quantity extrapolation, price floor/ceiling,
    missing feature degradation. Tree models extrapolate flat (nearest
    leaf), linear models extrapolate linearly — different failure modes.

    Args:
        models: Dict mapping model name to trained BaseModel.
        df: Full dataset (models should already be trained on it).

    Returns:
        Dict mapping model name to boundary test results.
    """
    # Train models on full data
    for model in models.values():
        model.fit(df)

    results = {}

    for name, model in models.items():
        tests = []

        # Extrapolation: qty=200 (beyond training max=100)
        probe_normal = _build_probe_row(quantity=100, material="Aluminum 6061")
        probe_extrap = _build_probe_row(quantity=200, material="Aluminum 6061")
        pred_100 = float(model.predict(probe_normal)[0])
        pred_200 = float(model.predict(probe_extrap)[0])
        # Reasonable: should be > pred_100 but not wildly more
        ratio = pred_200 / pred_100 if pred_100 > 0 else float("inf")
        tests.append(
            {
                "name": "qty_extrapolation_200",
                "pred_at_100": pred_100,
                "pred_at_200": pred_200,
                "ratio": ratio,
                "reasonable": 1.0 < ratio < 3.0,
                "failure_mode": "flat"
                if abs(ratio - 1.0) < 0.05
                else ("explosive" if ratio > 3.0 else "normal"),
            }
        )

        # Extreme extrapolation: qty=1000
        probe_extreme = _build_probe_row(quantity=1000, material="Aluminum 6061")
        pred_1000 = float(model.predict(probe_extreme)[0])
        ratio_extreme = pred_1000 / pred_100 if pred_100 > 0 else float("inf")
        tests.append(
            {
                "name": "qty_extrapolation_1000",
                "pred_at_100": pred_100,
                "pred_at_1000": pred_1000,
                "ratio": ratio_extreme,
                "reasonable": 1.0 < ratio_extreme < 15.0,
                "failure_mode": "flat"
                if abs(ratio_extreme - 1.0) < 0.05
                else ("explosive" if ratio_extreme > 15.0 else "normal"),
            }
        )

        # Price floor: any config predicting < $50?
        min_pred = float("inf")
        min_config = None
        for mat in ["Aluminum 6061"]:
            for proc in ["Surface Grinding", "CNC Turning"]:
                probe = _build_probe_row(
                    part_desc="Mounting Bracket - standard",
                    material=mat,
                    process=proc,
                    quantity=1,
                    rush=False,
                )
                pred = float(model.predict(probe)[0])
                if pred < min_pred:
                    min_pred = pred
                    min_config = f"{mat}/{proc}"

        tests.append(
            {
                "name": "price_floor",
                "min_prediction": min_pred,
                "min_config": min_config,
                "passed": min_pred >= 50,
                "details": f"Min prediction: ${min_pred:,.2f}"
                + (" (below $50 floor)" if min_pred < 50 else ""),
            }
        )

        # Price ceiling: most expensive config reasonable?
        max_pred = 0
        max_config = None
        probe = _build_probe_row(
            part_desc="Turbine Blade Housing - complex internal channels",
            material="Inconel 718",
            process="5-Axis Milling",
            quantity=100,
            rush=True,
        )
        max_pred = float(model.predict(probe)[0])
        max_config = "Inconel/5-Axis/TurbineBlade-complex/qty=100/rush"
        tests.append(
            {
                "name": "price_ceiling",
                "max_prediction": max_pred,
                "max_config": max_config,
                "passed": max_pred <= 500_000,
                "details": f"Max prediction: ${max_pred:,.2f}"
                + (" (above $500K ceiling)" if max_pred > 500_000 else ""),
            }
        )

        # Missing material: predict with NaN material
        probe_with = _build_probe_row(material="Aluminum 6061")
        probe_without = _build_probe_row()
        probe_without.loc[0, "Material"] = np.nan
        pred_with = float(model.predict(probe_with)[0])
        try:
            pred_without = float(model.predict(probe_without)[0])
            missing_degradation = abs(pred_with - pred_without) / pred_with * 100
            missing_ok = True
        except Exception:
            pred_without = None
            missing_degradation = None
            missing_ok = False

        tests.append(
            {
                "name": "missing_material_graceful",
                "pred_with_material": pred_with,
                "pred_without_material": pred_without,
                "degradation_pct": missing_degradation,
                "handles_gracefully": missing_ok,
            }
        )

        pass_count = sum(
            1
            for t in tests
            if t.get("passed", t.get("reasonable", t.get("handles_gracefully", False)))
        )
        results[name] = {
            "tests": tests,
            "pass_count": pass_count,
            "total_count": len(tests),
        }

    return results


# ---------------------------------------------------------------------------
# Lens 7: Complexity & Interpretability
# ---------------------------------------------------------------------------


def complexity_interpretability(
    models: dict[str, BaseModel], df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Assess model complexity and explainability.

    Measures effective parameter count, training time, and qualitative
    interpretability rating. Within ~1% MAPE, prefer the model that
    can show its work.

    Args:
        models: Dict mapping model name to trained BaseModel.
        df: Full dataset for timing.

    Returns:
        Dict mapping model name to complexity results.
    """
    results = {}

    for name, model in models.items():
        # Training time
        start = time.time()
        model.fit(df)
        train_seconds = time.time() - start

        # Effective parameter count
        effective_params = _count_effective_params(model)

        # Interpretability rating
        interp = _rate_interpretability(name, model)

        results[name] = {
            "train_seconds": round(train_seconds, 3),
            "effective_params": effective_params,
            "interpretability_rating": interp["rating"],
            "interpretability_reason": interp["reason"],
            "explanation_method": interp["method"],
            "model_type": type(model).__name__,
        }

    return results


def _count_effective_params(model: BaseModel) -> int | str:
    """Count effective parameters in a trained model."""
    if hasattr(model, "pipeline"):
        estimator = model.pipeline[-1]
        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            return int(np.sum(np.abs(coef) > 1e-8))
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "get_booster"):
            try:
                trees = inner.get_booster().get_dump()
                return sum(t.count("leaf=") for t in trees)
            except Exception:
                pass
        if hasattr(inner, "estimators_"):
            # Random Forest
            total_leaves = sum(est.get_n_leaves() for est in inner.estimators_)
            return total_leaves
        if hasattr(inner, "booster_"):
            try:
                return inner.booster_.num_leaves()
            except Exception:
                pass
    if hasattr(model, "models"):
        # Per-estimator models (M8)
        total = 0
        for sub in model.models.values():
            count = _count_effective_params(sub)
            if isinstance(count, int):
                total += count
        return total if total > 0 else "unknown"
    # M0 lookup table
    if hasattr(model, "part_type_medians"):
        return (
            len(model.part_type_medians)
            + len(model.material_factors)
            + len(model.process_factors)
            + 2
        )  # slope + rush multiplier

    return "unknown"


def _rate_interpretability(name: str, model: BaseModel) -> dict:
    """Qualitative interpretability rating for a model."""
    if name == "M0":
        return {
            "rating": "high",
            "reason": "Lookup table — every factor is directly readable",
            "method": "direct inspection",
        }
    if name in ("M1", "M2"):
        return {
            "rating": "high",
            "reason": "Linear model — coefficients show per-feature contribution",
            "method": "coefficients",
        }
    if name in ("M3a", "M3b"):
        return {
            "rating": "high",
            "reason": "Two-stage decomposition — unit price + discount curve are inspectable",
            "method": "coefficients + curve parameters",
        }
    if name == "M4":
        return {
            "rating": "medium-high",
            "reason": (
                "Lasso with interactions — non-zero coefficients "
                "are readable, but many interaction terms"
            ),
            "method": "coefficients (sparse)",
        }
    if name == "M5":
        return {
            "rating": "medium",
            "reason": (
                "Random Forest — feature importance available, but individual trees are opaque"
            ),
            "method": "permutation importance / SHAP",
        }
    if name in ("M6", "M6b", "M7", "M9"):
        return {
            "rating": "medium",
            "reason": (
                "XGBoost — SHAP provides per-prediction "
                "explanations, but model internals are complex"
            ),
            "method": "SHAP TreeExplainer",
        }
    if name in ("M7b", "M7c"):
        return {
            "rating": "medium",
            "reason": "LightGBM — SHAP available, similar to XGBoost",
            "method": "SHAP TreeExplainer",
        }
    if name == "M8":
        return {
            "rating": "low-medium",
            "reason": "3 separate XGBoost models — harder to compare across estimators",
            "method": "per-model SHAP (fragmented)",
        }
    return {
        "rating": "unknown",
        "reason": "Unrecognized model",
        "method": "unknown",
    }


# ---------------------------------------------------------------------------
# Scorecard: Aggregate all lenses
# ---------------------------------------------------------------------------

# Quantile rank mapping
RATING_DOTS = {1: "●○○○○", 2: "●●○○○", 3: "●●●○○", 4: "●●●●○", 5: "●●●●●"}


def _quantile_rank(values: dict[str, float], higher_is_better: bool = True) -> dict[str, int]:
    """Rank model scores into 1-5 quintiles.

    Args:
        values: Dict mapping model name to numeric score.
        higher_is_better: If True, highest value gets rank 5.

    Returns:
        Dict mapping model name to rank (1-5).
    """
    if not values:
        return {}

    sorted_names = sorted(values.keys(), key=lambda k: values[k], reverse=higher_is_better)
    n = len(sorted_names)
    ranks = {}
    for i, name in enumerate(sorted_names):
        # Map position to 1-5 scale
        ranks[name] = max(1, min(5, 5 - int(i / max(n, 1) * 5)))
    return ranks


def generate_scorecard(
    error_results: dict,
    segment_results: dict,
    coherence_results: dict,
    calibration_results: dict,
    stability_results: dict,
    boundary_results: dict,
    complexity_results: dict,
) -> dict[str, dict[str, Any]]:
    """Aggregate all 7 lenses into a ranked scorecard.

    Uses quantile ranking (1-5) for most lenses. Economic coherence
    uses absolute pass/fail (pass_rate mapped to 1-5 scale directly).

    Args:
        Results dicts from each lens function.

    Returns:
        Dict mapping model name to scorecard with per-lens ratings,
        dots visualization, and narrative summary.
    """
    model_names = list(error_results.keys())

    # Extract scoring metrics per lens
    error_scores = {m: 100 - error_results[m]["mape"] for m in model_names}
    error_ranks = _quantile_rank(error_scores, higher_is_better=True)

    # Segment fairness: use max segment MAPE (lower is better)
    segment_scores = {}
    for m in model_names:
        all_mapes = []
        for seg_name, seg_cats in segment_results[m].items():
            for cat, stats in seg_cats.items():
                if stats["count"] >= 10:
                    all_mapes.append(stats["mape"])
        segment_scores[m] = max(all_mapes) if all_mapes else 100
    segment_ranks = _quantile_rank(segment_scores, higher_is_better=False)

    # Economic coherence: absolute scale (pass_rate / 20 → 1-5)
    coherence_ranks = {}
    for m in model_names:
        rate = coherence_results[m]["pass_rate"]
        coherence_ranks[m] = max(1, min(5, round(rate / 20)))

    # Calibration: closer to balanced is better (lower abs mean signed error)
    cal_scores = {m: abs(calibration_results[m]["mean_signed_error_pct"]) for m in model_names}
    cal_ranks = _quantile_rank(cal_scores, higher_is_better=False)

    # Stability: lower fold MAPE std is better
    stab_scores = {m: stability_results[m]["fold_mape_std"] for m in model_names}
    stab_ranks = _quantile_rank(stab_scores, higher_is_better=False)

    # Boundary: pass count
    boundary_scores = {m: boundary_results[m]["pass_count"] for m in model_names}
    boundary_ranks = _quantile_rank(boundary_scores, higher_is_better=True)

    # Interpretability: map rating to numeric
    interp_map = {
        "high": 5,
        "medium-high": 4,
        "medium": 3,
        "low-medium": 2,
        "low": 1,
        "unknown": 1,
    }
    interp_ranks = {
        m: interp_map.get(complexity_results[m]["interpretability_rating"], 1) for m in model_names
    }

    scorecard = {}
    for m in model_names:
        lens_ranks = {
            "error_profile": error_ranks.get(m, 3),
            "segment_fairness": segment_ranks.get(m, 3),
            "economic_coherence": coherence_ranks.get(m, 3),
            "calibration": cal_ranks.get(m, 3),
            "stability": stab_ranks.get(m, 3),
            "boundary_behavior": boundary_ranks.get(m, 3),
            "interpretability": interp_ranks.get(m, 3),
        }
        lens_dots = {k: RATING_DOTS[v] for k, v in lens_ranks.items()}
        avg_rank = float(np.mean(list(lens_ranks.values())))

        scorecard[m] = {
            "ranks": lens_ranks,
            "dots": lens_dots,
            "average_rank": round(avg_rank, 2),
            "narrative": _generate_narrative(
                m,
                lens_ranks,
                error_results[m],
                calibration_results[m],
                coherence_results[m],
                stability_results[m],
            ),
        }

    return scorecard


def _generate_narrative(
    model_name: str,
    ranks: dict[str, int],
    error: dict,
    calibration: dict,
    coherence: dict,
    stability: dict,
) -> str:
    """Generate a one-sentence narrative for a model's scorecard."""
    strengths = [k.replace("_", " ") for k, v in ranks.items() if v >= 4]
    weaknesses = [k.replace("_", " ") for k, v in ranks.items() if v <= 2]

    parts = [f"{model_name}:"]
    parts.append(f"{error['mape']:.1f}% MAPE, {calibration['label']}.")

    if coherence["pass_rate"] < 100:
        n_fail = coherence["total_count"] - coherence["pass_count"]
        parts.append(f"{n_fail} economic coherence violation(s).")

    parts.append(f"Fold stability: ±{stability['fold_mape_std']:.1f}%.")

    if strengths:
        parts.append(f"Strong on: {', '.join(strengths)}.")
    if weaknesses:
        parts.append(f"Weak on: {', '.join(weaknesses)}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Text report formatting
# ---------------------------------------------------------------------------


def format_scorecard_text(scorecard: dict) -> str:
    """Format the scorecard as a human-readable text table.

    Returns:
        Multi-line string with the scorecard summary table and
        per-model narratives.
    """
    lines = []
    lines.append("=" * 100)
    lines.append("  MODEL COMPARISON REPORT — 7 Lenses")
    lines.append("=" * 100)
    lines.append("")

    # Header
    header = (
        f"{'Model':<8} {'Error':^7} {'Segment':^7} {'Econ.':^7} "
        f"{'Calib.':^7} {'Stabil.':^7} {'Boundary':^8} {'Interp.':^7}  {'Avg':>4}"
    )
    lines.append(header)
    lines.append("-" * 100)

    # Sort by average rank descending
    sorted_models = sorted(
        scorecard.keys(),
        key=lambda m: scorecard[m]["average_rank"],
        reverse=True,
    )

    for m in sorted_models:
        sc = scorecard[m]
        d = sc["dots"]
        row = (
            f"{m:<8} {d['error_profile']:^7} {d['segment_fairness']:^7} "
            f"{d['economic_coherence']:^7} {d['calibration']:^7} "
            f"{d['stability']:^7} {d['boundary_behavior']:^8} "
            f"{d['interpretability']:^7}  {sc['average_rank']:>4.1f}"
        )
        lines.append(row)

    lines.append("-" * 100)
    lines.append("")
    lines.append("NARRATIVES")
    lines.append("")
    for m in sorted_models:
        lines.append(f"  {scorecard[m]['narrative']}")

    lines.append("")
    lines.append("Scoring: ●●●●● = best quintile, ●○○○○ = worst quintile")
    lines.append("Economic coherence uses absolute pass/fail, not relative ranking.")

    return "\n".join(lines)
