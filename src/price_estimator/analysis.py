"""EDA analysis functions for the price estimator.

Computes summary statistics, volume discount curves, rush premiums,
lead time correlations, and confounding checks. All functions return
plain dicts suitable for JSON serialization.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from price_estimator.data import compute_unit_price

logger = logging.getLogger(__name__)


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Compute basic dataset summary: row count, column types, missing counts.

    Args:
        df: Cleaned DataFrame from data.load_data().

    Returns:
        Dict with row_count, column_count, column_types, missing_counts,
        and target_stats (mean, median, std, min, max).
    """
    col_types = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "datetime" in dtype:
            col_types[col] = "datetime"
        elif "bool" in dtype:
            col_types[col] = "boolean"
        elif "int" in dtype:
            col_types[col] = "integer"
        elif "float" in dtype:
            col_types[col] = "float"
        else:
            col_types[col] = "object"

    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "column_types": col_types,
        "missing_counts": missing_counts,
        "target_stats": {
            "mean": float(df["TotalPrice_USD"].mean()),
            "median": float(df["TotalPrice_USD"].median()),
            "std": float(df["TotalPrice_USD"].std()),
            "min": float(df["TotalPrice_USD"].min()),
            "max": float(df["TotalPrice_USD"].max()),
        },
    }


def compute_unit_price_analysis(df: pd.DataFrame) -> dict:
    """Compute unit price summary stats grouped by material, part type, process, estimator.

    Args:
        df: Cleaned DataFrame from data.load_data().

    Returns:
        Dict with keys by_material, by_part_type, by_process, by_estimator.
        Each value is a dict mapping category values to {mean, median, std, count}.
    """
    unit_price = compute_unit_price(df)

    def _group_stats(groupby_col: str) -> dict:
        grouped = df.assign(unit_price=unit_price).groupby(groupby_col)["unit_price"]
        result = {}
        for name, group in grouped:
            result[name] = {
                "mean": float(group.mean()),
                "median": float(group.median()),
                "std": float(group.std()),
                "count": int(group.count()),
            }
        return result

    return {
        "by_material": _group_stats("Material"),
        "by_part_type": _group_stats("PartDescription"),
        "by_process": _group_stats("Process"),
        "by_estimator": _group_stats("Estimator"),
    }


def compute_volume_discount(df: pd.DataFrame) -> dict:
    """Fit log(qty) vs log(unit_price) globally and per part type.

    Args:
        df: Cleaned DataFrame from data.load_data().

    Returns:
        Dict with global regression stats (slope, intercept, r_squared, etc.)
        and per_part_type dict mapping each base part type to its own
        regression stats.
    """
    unit_price = compute_unit_price(df)
    log_qty = np.log(df["Quantity"].astype(float))
    log_up = np.log(unit_price)

    # Global fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_qty, log_up)

    result = {
        "log_qty_vs_log_unit_price_correlation": float(r_value),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "std_err": float(std_err),
    }

    # Per part type fit
    base_types = df["PartDescription"].str.split(" - ").str[0]
    per_part = {}
    for pt in sorted(base_types.unique()):
        mask = base_types == pt
        if mask.sum() < 3:
            continue
        pt_slope, pt_intercept, pt_r, pt_p, pt_se = stats.linregress(log_qty[mask], log_up[mask])
        per_part[pt] = {
            "slope": float(pt_slope),
            "intercept": float(pt_intercept),
            "r_squared": float(pt_r**2),
            "n": int(mask.sum()),
        }

    result["per_part_type"] = per_part
    return result


def compute_rush_premium(df: pd.DataFrame) -> dict:
    """Compute rush premium: overall controlled ratio and breakdowns by part type and material.

    Controls for part type, material, and process by computing the ratio
    within each (PartDescription, Material, Process) group, then averaging.

    Args:
        df: Cleaned DataFrame from data.load_data().

    Returns:
        Dict with controlled_mean_ratio, marginal_ratio, and breakdowns
        by_part_type and by_material showing per-category rush ratios.
    """
    unit_price = compute_unit_price(df)
    analysis_df = df.assign(unit_price=unit_price).copy()

    # Drop rows with missing Material or Process for controlled comparison
    controlled = analysis_df.dropna(subset=["Material", "Process"])

    # Group by part/material/process and compute mean unit price for rush vs non-rush
    group_cols = ["PartDescription", "Material", "Process"]
    rush_means = controlled[controlled["RushJob"]].groupby(group_cols)["unit_price"].mean()
    no_rush_means = controlled[~controlled["RushJob"]].groupby(group_cols)["unit_price"].mean()

    # Only keep groups that have both rush and non-rush observations
    common_groups = rush_means.index.intersection(no_rush_means.index)
    if len(common_groups) > 0:
        ratios = rush_means.loc[common_groups] / no_rush_means.loc[common_groups]
        controlled_ratio = float(ratios.mean())
        n_groups = len(common_groups)
    else:
        controlled_ratio = None
        n_groups = 0

    # Marginal (uncontrolled) comparison
    rush_marginal = float(analysis_df[analysis_df["RushJob"]]["unit_price"].mean())
    no_rush_marginal = float(analysis_df[~analysis_df["RushJob"]]["unit_price"].mean())
    marginal_ratio = rush_marginal / no_rush_marginal

    # Breakdown by part type
    base_types = analysis_df["PartDescription"].str.split(" - ").str[0]
    analysis_df = analysis_df.assign(base_part_type=base_types)
    by_part_type = {}
    for pt in sorted(base_types.unique()):
        pt_df = analysis_df[analysis_df["base_part_type"] == pt]
        rush_up = pt_df[pt_df["RushJob"]]["unit_price"]
        no_rush_up = pt_df[~pt_df["RushJob"]]["unit_price"]
        if len(rush_up) > 0 and len(no_rush_up) > 0:
            by_part_type[pt] = {
                "ratio": float(rush_up.mean() / no_rush_up.mean()),
                "n_rush": int(len(rush_up)),
                "n_no_rush": int(len(no_rush_up)),
            }

    # Breakdown by material
    by_material = {}
    for mat in sorted(analysis_df["Material"].dropna().unique()):
        mat_df = analysis_df[analysis_df["Material"] == mat]
        rush_up = mat_df[mat_df["RushJob"]]["unit_price"]
        no_rush_up = mat_df[~mat_df["RushJob"]]["unit_price"]
        if len(rush_up) > 0 and len(no_rush_up) > 0:
            by_material[mat] = {
                "ratio": float(rush_up.mean() / no_rush_up.mean()),
                "n_rush": int(len(rush_up)),
                "n_no_rush": int(len(no_rush_up)),
            }

    return {
        "controlled_mean_ratio": controlled_ratio,
        "n_controlled_groups": n_groups,
        "marginal_ratio": float(marginal_ratio),
        "rush_mean_unit_price": rush_marginal,
        "no_rush_mean_unit_price": no_rush_marginal,
        "by_part_type": by_part_type,
        "by_material": by_material,
    }


def compute_lead_time_analysis(df: pd.DataFrame) -> dict:
    """Compute lead time vs unit price correlation and means by bucket.

    Args:
        df: Cleaned DataFrame from data.load_data().

    Returns:
        Dict with correlation, p_value, and by_bucket stats
        (short 2-4, medium 5-8, long 9-12).
    """
    unit_price = compute_unit_price(df)
    corr, p_value = stats.pearsonr(df["LeadTimeWeeks"], unit_price)

    bins = [0, 4, 8, 12]
    labels = ["short (2-4)", "medium (5-8)", "long (9-12)"]
    buckets = pd.cut(df["LeadTimeWeeks"], bins=bins, labels=labels)
    bucket_means = (
        df.assign(unit_price=unit_price, bucket=buckets)
        .groupby("bucket", observed=False)["unit_price"]
        .agg(["mean", "median", "count"])
    )
    by_bucket = {}
    for bucket_name, row in bucket_means.iterrows():
        by_bucket[str(bucket_name)] = {
            "mean": float(row["mean"]),
            "median": float(row["median"]),
            "count": int(row["count"]),
        }

    return {
        "correlation_with_unit_price": float(corr),
        "p_value": float(p_value),
        "by_bucket": by_bucket,
    }


def compute_confounding_analysis(df: pd.DataFrame) -> dict:
    """Run confounding checks: chi-squared, conditional means, propensity classifier.

    Tests whether estimator assignment is correlated with job characteristics,
    which would confound the bias analysis.

    Args:
        df: Cleaned DataFrame from data.load_data().

    Returns:
        Dict with chi_squared_tests, marginal and conditional mean unit prices
        by estimator, and propensity classifier results.
    """
    unit_price = compute_unit_price(df)
    analysis_df = df.assign(unit_price=unit_price)

    # 1. Chi-squared tests for independence
    chi2_results = {}
    pairs = [
        ("Estimator", "PartDescription"),
        ("Estimator", "Material"),
        ("Estimator", "Process"),
        ("RushJob", "Estimator"),
    ]
    for col_a, col_b in pairs:
        subset = analysis_df.dropna(subset=[col_a, col_b])
        ct = pd.crosstab(subset[col_a], subset[col_b])
        chi2, p, dof, _expected = stats.chi2_contingency(ct)
        pair_key = f"{col_a}_x_{col_b}"
        chi2_results[pair_key] = {
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "significant_at_005": bool(p < 0.05),
        }

    # 2. Conditional vs marginal mean unit prices by estimator
    marginal_means = analysis_df.groupby("Estimator")["unit_price"].mean().to_dict()

    cond_df = analysis_df.dropna(subset=["Material"])
    group_means = cond_df.groupby(["PartDescription", "Material"])["unit_price"].transform("mean")
    cond_df = cond_df.assign(residual=cond_df["unit_price"] - group_means)
    conditional_adjustments = cond_df.groupby("Estimator")["residual"].mean().to_dict()
    grand_mean = float(cond_df["unit_price"].mean())
    conditional_means = {
        est: float(grand_mean + adj) for est, adj in conditional_adjustments.items()
    }

    # 3. Propensity classifier: can we predict estimator from job features?
    prop_df = analysis_df.dropna(subset=["Material", "Process"]).copy()
    le_target = LabelEncoder()
    y_prop = le_target.fit_transform(prop_df["Estimator"])

    feature_cols_cat = ["Material", "Process", "PartDescription"]
    X_prop = pd.DataFrame(index=prop_df.index)
    for col in feature_cols_cat:
        le = LabelEncoder()
        X_prop[col] = le.fit_transform(prop_df[col])
    X_prop["Quantity"] = prop_df["Quantity"].values
    X_prop["RushJob"] = prop_df["RushJob"].astype(int).values
    X_prop["LeadTimeWeeks"] = prop_df["LeadTimeWeeks"].values

    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
    cv_scores = cross_val_score(clf, X_prop, y_prop, cv=5, scoring="accuracy")
    propensity_accuracy = float(cv_scores.mean())
    random_baseline = 1.0 / len(le_target.classes_)

    return {
        "chi_squared_tests": chi2_results,
        "marginal_mean_unit_price_by_estimator": {k: float(v) for k, v in marginal_means.items()},
        "conditional_mean_unit_price_by_estimator": conditional_means,
        "propensity_classifier": {
            "accuracy": propensity_accuracy,
            "cv_scores": [float(s) for s in cv_scores],
            "random_baseline": float(random_baseline),
            "non_random_assignment": propensity_accuracy > 0.40,
        },
    }
