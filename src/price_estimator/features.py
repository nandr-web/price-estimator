"""Feature engineering for the price estimator.

This module provides the Tier 1 PartDescription parser (deterministic
dict registry with rapidfuzz fallback for typo tolerance) and feature
matrix construction for both linear models (one-hot encoding) and tree
models (label encoding).
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier 1 PartDescription Registry
# ---------------------------------------------------------------------------

# Maps known base part types to their canonical names.
# Matching is done on the text before the " - " separator.
BASE_PART_TYPES = {
    "Actuator Linkage": "Actuator Linkage",
    "Electronic Chassis": "Electronic Chassis",
    "Fuel Injector Nozzle": "Fuel Injector Nozzle",
    "Heat Sink": "Heat Sink",
    "Landing Gear Pin": "Landing Gear Pin",
    "Manifold Block": "Manifold Block",
    "Mounting Bracket": "Mounting Bracket",
    "Sensor Housing": "Sensor Housing",
    "Structural Rib": "Structural Rib",
    "Turbine Blade Housing": "Turbine Blade Housing",
}

# Maps modifier text (after the " - " separator) to a complexity flag name.
# Each modifier also has a complexity weight used to compute the aggregate score.
MODIFIER_REGISTRY: dict[str, tuple[str, int]] = {
    "thin walls": ("thin_walls", 2),
    "complex internal channels": ("complex_internal", 3),
    "high precision": ("high_precision", 2),
    "hardened": ("hardened", 1),
    "threaded": ("threaded", 1),
    "aerospace grade": ("aerospace_grade", 1),
    "EMI shielded": ("emi_shielded", 1),
    "high fin density": ("high_fin_density", 2),
    "standard": ("standard", 0),
}

ALL_COMPLEXITY_FLAGS = [flag for flag, _weight in MODIFIER_REGISTRY.values()]

# Minimum rapidfuzz similarity score (0-100) to accept a fuzzy match.
FUZZY_MATCH_THRESHOLD = 85

# ---------------------------------------------------------------------------
# Material and process ordinal tiers
# ---------------------------------------------------------------------------

# Ordered by machinability difficulty / cost (higher = more expensive).
MATERIAL_COST_TIER = {
    "Aluminum 6061": 1,
    "Aluminum 7075": 2,
    "Stainless Steel 17-4 PH": 3,
    "Titanium Grade 5": 4,
    "Inconel 718": 5,
}

# Ordered by machine capability / cost (higher = more expensive).
PROCESS_PRECISION_TIER = {
    "Surface Grinding": 1,
    "CNC Turning": 2,
    "3-Axis Milling": 3,
    "Wire EDM": 4,
    "5-Axis Milling": 5,
}


# ---------------------------------------------------------------------------
# Parser result
# ---------------------------------------------------------------------------


@dataclass
class ParsedDescription:
    """Result of parsing a PartDescription string.

    Attributes:
        base_type: Canonical base part type name, or "UNKNOWN" if unrecognized.
        modifiers: Dict mapping complexity flag names to True/False.
        complexity_score: Weighted sum of active modifiers.
        fuzzy_matched: Whether the base type required fuzzy matching.
        match_score: Similarity score of the fuzzy match (100 for exact match).
    """

    base_type: str = "UNKNOWN"
    modifiers: dict[str, bool] = field(default_factory=dict)
    complexity_score: int = 0
    fuzzy_matched: bool = False
    match_score: float = 100.0

    def __post_init__(self):
        # Ensure all flags are present, defaulting to False
        for flag in ALL_COMPLEXITY_FLAGS:
            self.modifiers.setdefault(flag, False)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_part_description(description: str) -> ParsedDescription:
    """Parse a PartDescription into a base type and complexity flags.

    Uses the Tier 1 deterministic registry with rapidfuzz fallback for
    typo tolerance. Splits on " - " to separate the base part type from
    modifiers. Each modifier is matched against the modifier registry.

    Args:
        description: Raw PartDescription string from the CSV,
            e.g. "Sensor Housing - threaded" or "Actuator Linkage".

    Returns:
        ParsedDescription with base_type, modifier flags, and complexity_score.

    Examples:
        >>> result = parse_part_description("Sensor Housing - threaded")
        >>> result.base_type
        'Sensor Housing'
        >>> result.modifiers["threaded"]
        True
        >>> result.complexity_score
        1

        >>> result = parse_part_description("Actuator Linkage")
        >>> result.base_type
        'Actuator Linkage'
        >>> result.complexity_score
        0
    """
    result = ParsedDescription()

    # Split base type from modifiers
    if " - " in description:
        base_raw, modifier_raw = description.split(" - ", maxsplit=1)
    else:
        base_raw = description
        modifier_raw = ""

    base_raw = base_raw.strip()
    modifier_raw = modifier_raw.strip()

    # --- Match base type ---
    if base_raw in BASE_PART_TYPES:
        result.base_type = BASE_PART_TYPES[base_raw]
        result.match_score = 100.0
    else:
        # Fuzzy match against known base types
        match = process.extractOne(
            base_raw,
            BASE_PART_TYPES.keys(),
            scorer=fuzz.ratio,
            score_cutoff=FUZZY_MATCH_THRESHOLD,
        )
        if match is not None:
            matched_key, score, _index = match
            result.base_type = BASE_PART_TYPES[matched_key]
            result.fuzzy_matched = True
            result.match_score = score
            logger.info("Fuzzy matched '%s' -> '%s' (score=%.1f)", base_raw, matched_key, score)
        else:
            result.base_type = "UNKNOWN"
            result.fuzzy_matched = True
            result.match_score = 0.0
            logger.warning("Could not match base type: '%s'", base_raw)

    # --- Match modifiers ---
    if modifier_raw:
        # Modifiers can be comma-separated (future-proofing)
        modifier_parts = [m.strip() for m in modifier_raw.split(",")]
        for mod_text in modifier_parts:
            if mod_text in MODIFIER_REGISTRY:
                flag_name, weight = MODIFIER_REGISTRY[mod_text]
                result.modifiers[flag_name] = True
                result.complexity_score += weight
            else:
                # Fuzzy match modifier
                mod_match = process.extractOne(
                    mod_text,
                    MODIFIER_REGISTRY.keys(),
                    scorer=fuzz.ratio,
                    score_cutoff=FUZZY_MATCH_THRESHOLD,
                )
                if mod_match is not None:
                    matched_mod, score, _index = mod_match
                    flag_name, weight = MODIFIER_REGISTRY[matched_mod]
                    result.modifiers[flag_name] = True
                    result.complexity_score += weight
                    logger.info(
                        "Fuzzy matched modifier '%s' -> '%s' (score=%.1f)",
                        mod_text,
                        matched_mod,
                        score,
                    )
                else:
                    logger.warning("Unknown modifier: '%s'", mod_text)

    return result


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------


def extract_description_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse all PartDescription values and return a features DataFrame.

    Applies parse_part_description() to each row and expands the results
    into columns: base_part_type, one boolean column per complexity flag,
    and complexity_score.

    Args:
        df: DataFrame with a PartDescription column.

    Returns:
        DataFrame with columns: base_part_type, <all complexity flags>,
        complexity_score. Same index as input.
    """
    parsed = df["PartDescription"].apply(parse_part_description)

    records = []
    for p in parsed:
        row = {"base_part_type": p.base_type, "complexity_score": p.complexity_score}
        for flag in ALL_COMPLEXITY_FLAGS:
            row[flag] = p.modifiers.get(flag, False)
        records.append(row)

    return pd.DataFrame(records, index=df.index)


def build_feature_matrix(
    df: pd.DataFrame, encoding: str = "onehot"
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the complete feature matrix from the cleaned DataFrame.

    Constructs all features needed for model training: parsed PartDescription
    features, material/process encoding, log quantity, rush job flag,
    lead time, and optional ordinal tiers.

    Args:
        df: Cleaned DataFrame from data.load_data().
        encoding: Feature encoding strategy.
            "onehot" — one-hot encode categoricals (for linear models).
            "label" — label-encode categoricals (for tree models).

    Returns:
        Tuple of (X, y) where X is the feature matrix and y is TotalPrice_USD.

    Raises:
        ValueError: If encoding is not "onehot" or "label".
    """
    if encoding not in ("onehot", "label"):
        raise ValueError(f"encoding must be 'onehot' or 'label', got '{encoding}'")

    # Start with description-derived features
    desc_features = extract_description_features(df)

    # Numeric features
    features = pd.DataFrame(index=df.index)
    features["log_quantity"] = np.log(df["Quantity"].astype(float))
    features["rush_job"] = df["RushJob"].astype(int)
    features["lead_time_weeks"] = df["LeadTimeWeeks"]

    # Ordinal tiers (useful for both encoding types)
    features["material_cost_tier"] = df["Material"].map(MATERIAL_COST_TIER).astype(float)
    features["process_precision_tier"] = df["Process"].map(PROCESS_PRECISION_TIER).astype(float)

    # Complexity features
    for flag in ALL_COMPLEXITY_FLAGS:
        features[flag] = desc_features[flag].astype(int)
    features["complexity_score"] = desc_features["complexity_score"]

    # Categorical features
    categoricals = {
        "base_part_type": desc_features["base_part_type"],
        "material": df["Material"],
        "process": df["Process"],
        "estimator": df["Estimator"],
    }

    if encoding == "onehot":
        for name, series in categoricals.items():
            dummies = pd.get_dummies(series, prefix=name, dtype=int)
            features = pd.concat([features, dummies], axis=1)
    elif encoding == "label":
        for name, series in categoricals.items():
            features[name] = series.astype("category").cat.codes

    y = df["TotalPrice_USD"]
    return features, y


def build_feature_matrix_no_estimator(
    df: pd.DataFrame, encoding: str = "onehot"
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix without the Estimator feature.

    Used for the debiased model (M9) which excludes estimator to compute
    neutral prices. Identical to build_feature_matrix() except the
    estimator column is omitted.

    Args:
        df: Cleaned DataFrame from data.load_data().
        encoding: "onehot" or "label".

    Returns:
        Tuple of (X, y) with estimator columns excluded.
    """
    X, y = build_feature_matrix(df, encoding=encoding)

    # Drop estimator-related columns
    estimator_cols = [c for c in X.columns if c.startswith("estimator")]
    X = X.drop(columns=estimator_cols)

    return X, y
