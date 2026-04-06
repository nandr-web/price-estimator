"""Data loading, validation, and profiling for the price estimator.

This module handles CSV ingestion, schema validation, and missing value
analysis. It does NOT perform imputation — that belongs inside sklearn
Pipelines during cross-validation to prevent data leakage.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "QuoteID",
    "Date",
    "PartDescription",
    "Material",
    "Process",
    "Quantity",
    "LeadTimeWeeks",
    "RushJob",
    "Estimator",
    "TotalPrice_USD",
]

VALID_MATERIALS = [
    "Aluminum 6061",
    "Aluminum 7075",
    "Inconel 718",
    "Stainless Steel 17-4 PH",
    "Titanium Grade 5",
]

VALID_PROCESSES = [
    "3-Axis Milling",
    "5-Axis Milling",
    "CNC Turning",
    "Surface Grinding",
    "Wire EDM",
]

VALID_ESTIMATORS = ["Sato-san", "Suzuki-san", "Tanaka-san"]

VALID_QUANTITIES = [1, 5, 10, 20, 50, 100]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the historical quotes CSV and apply basic type conversions.

    Reads the CSV, converts Date to datetime, RushJob to boolean, and
    replaces empty strings with NaN for proper missing value handling.
    Does not impute missing values.

    Args:
        path: Path to the aora_historical_quotes.csv file.

    Returns:
        DataFrame with proper dtypes and missing values as NaN.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    # Check required columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Replace empty strings with NaN for consistent missing value handling
    df = df.replace("", np.nan)

    # Type conversions
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["RushJob"] = df["RushJob"].map({"Yes": True, "No": False})
    df["Quantity"] = df["Quantity"].astype(int)
    df["LeadTimeWeeks"] = df["LeadTimeWeeks"].astype(int)
    df["TotalPrice_USD"] = df["TotalPrice_USD"].astype(float)

    return df


def validate(df: pd.DataFrame) -> list[str]:
    """Run schema and data quality validations on the loaded DataFrame.

    Checks value ranges, categorical levels, and data integrity constraints.
    Returns a list of warning messages for any issues found. Raises on
    critical violations (negative prices, invalid rush job values).

    Args:
        df: DataFrame from load_data().

    Returns:
        List of warning strings describing data quality issues.
        Empty list if no issues found.

    Raises:
        ValueError: If critical data integrity violations are found
            (e.g., negative prices, NaN in RushJob).
    """
    warnings = []

    # Critical checks — raise on failure
    if df["TotalPrice_USD"].le(0).any():
        raise ValueError("Found non-positive prices in TotalPrice_USD")

    if df["RushJob"].isna().any():
        raise ValueError("Found NaN values in RushJob after type conversion")

    # Quantity validation
    invalid_qty = ~df["Quantity"].isin(VALID_QUANTITIES)
    if invalid_qty.any():
        bad = df.loc[invalid_qty, "Quantity"].unique().tolist()
        raise ValueError(f"Invalid Quantity values: {bad}. Expected one of {VALID_QUANTITIES}")

    # Warning checks — log but don't raise
    n_missing_material = df["Material"].isna().sum()
    if n_missing_material > 0:
        warnings.append(f"{n_missing_material} rows with missing Material")
        logger.warning(warnings[-1])

    n_missing_process = df["Process"].isna().sum()
    if n_missing_process > 0:
        warnings.append(f"{n_missing_process} rows with missing Process")
        logger.warning(warnings[-1])

    # Check for unexpected categorical values
    known_materials = df["Material"].dropna()
    unknown_materials = set(known_materials) - set(VALID_MATERIALS)
    if unknown_materials:
        warnings.append(f"Unknown Material values: {unknown_materials}")
        logger.warning(warnings[-1])

    known_processes = df["Process"].dropna()
    unknown_processes = set(known_processes) - set(VALID_PROCESSES)
    if unknown_processes:
        warnings.append(f"Unknown Process values: {unknown_processes}")
        logger.warning(warnings[-1])

    unknown_estimators = set(df["Estimator"]) - set(VALID_ESTIMATORS)
    if unknown_estimators:
        warnings.append(f"Unknown Estimator values: {unknown_estimators}")
        logger.warning(warnings[-1])

    # Duplicate QuoteIDs
    n_dupes = df["QuoteID"].duplicated().sum()
    if n_dupes > 0:
        warnings.append(f"{n_dupes} duplicate QuoteID values")
        logger.warning(warnings[-1])

    # Lead time range
    if df["LeadTimeWeeks"].min() < 1:
        warnings.append(f"LeadTimeWeeks has values < 1: min={df['LeadTimeWeeks'].min()}")
        logger.warning(warnings[-1])

    return warnings


def get_missing_report(df: pd.DataFrame) -> dict:
    """Analyze missing value patterns in the dataset.

    Checks which columns have missing values, whether missing Material and
    Process rows overlap, and whether missingness correlates with other
    features (estimator, part type, rush job).

    Args:
        df: DataFrame from load_data().

    Returns:
        Dictionary containing:
            - counts: dict of column -> number of missing values
            - missing_material_ids: list of QuoteIDs with missing Material
            - missing_process_ids: list of QuoteIDs with missing Process
            - overlap_count: number of rows missing both Material and Process
            - overlap_ids: list of QuoteIDs missing both
            - by_estimator: dict of estimator -> count of rows with any missing
            - by_part_type: dict of part type -> count of rows with any missing
    """
    counts = {}
    for col in df.columns:
        n = df[col].isna().sum()
        if n > 0:
            counts[col] = int(n)

    missing_material = df[df["Material"].isna()]
    missing_process = df[df["Process"].isna()]

    material_ids = missing_material["QuoteID"].tolist()
    process_ids = missing_process["QuoteID"].tolist()
    overlap_ids = sorted(set(material_ids) & set(process_ids))

    # Check if missingness correlates with other features
    any_missing = df["Material"].isna() | df["Process"].isna()
    rows_with_missing = df[any_missing]

    by_estimator = rows_with_missing["Estimator"].value_counts().to_dict()
    by_part_type = rows_with_missing["PartDescription"].value_counts().to_dict()

    return {
        "counts": counts,
        "missing_material_ids": material_ids,
        "missing_process_ids": process_ids,
        "overlap_count": len(overlap_ids),
        "overlap_ids": overlap_ids,
        "by_estimator": by_estimator,
        "by_part_type": by_part_type,
    }


def compute_unit_price(df: pd.DataFrame) -> pd.Series:
    """Compute per-unit price as TotalPrice_USD / Quantity.

    Args:
        df: DataFrame with TotalPrice_USD and Quantity columns.

    Returns:
        Series of unit prices, same index as input.
    """
    return df["TotalPrice_USD"] / df["Quantity"]
