"""Tests for feature engineering: PartDescription parser and feature matrix."""

import numpy as np
import pandas as pd
import pytest

from price_estimator.features import (
    ALL_COMPLEXITY_FLAGS,
    build_feature_matrix,
    build_feature_matrix_no_estimator,
    extract_description_features,
    parse_part_description,
)


class TestParsePartDescription:
    """Tests for the Tier 1 PartDescription parser."""

    @pytest.mark.parametrize(
        "description,expected_base,expected_flag,expected_score",
        [
            ("Sensor Housing - threaded", "Sensor Housing", "threaded", 1),
            (
                "Manifold Block - complex internal channels",
                "Manifold Block",
                "complex_internal",
                3,
            ),
            ("Mounting Bracket - standard", "Mounting Bracket", "standard", 0),
            ("Turbine Blade Housing - thin walls", "Turbine Blade Housing", "thin_walls", 2),
            (
                "Fuel Injector Nozzle - high precision",
                "Fuel Injector Nozzle",
                "high_precision",
                2,
            ),
            ("Landing Gear Pin - hardened", "Landing Gear Pin", "hardened", 1),
            ("Heat Sink - high fin density", "Heat Sink", "high_fin_density", 2),
            (
                "Electronic Chassis - EMI shielded",
                "Electronic Chassis",
                "emi_shielded",
                1,
            ),
            ("Structural Rib - aerospace grade", "Structural Rib", "aerospace_grade", 1),
        ],
    )
    def test_parse_all_known_types_with_modifiers(
        self, description, expected_base, expected_flag, expected_score
    ):
        """All 9 part types with modifiers parse correctly."""
        result = parse_part_description(description)
        assert result.base_type == expected_base
        assert result.modifiers[expected_flag] is True
        assert result.complexity_score == expected_score
        assert not result.fuzzy_matched

    def test_parse_no_modifier(self):
        """Part type with no modifier parses correctly."""
        result = parse_part_description("Actuator Linkage")
        assert result.base_type == "Actuator Linkage"
        assert result.complexity_score == 0
        assert all(v is False for v in result.modifiers.values())
        assert not result.fuzzy_matched

    def test_all_flags_present_in_result(self):
        """All known complexity flags are present in every parse result."""
        result = parse_part_description("Actuator Linkage")
        for flag in ALL_COMPLEXITY_FLAGS:
            assert flag in result.modifiers

    def test_parse_fuzzy_base_type(self):
        """Fuzzy matching catches typos in base part type."""
        result = parse_part_description("Sensor Housng - threaded")
        assert result.base_type == "Sensor Housing"
        assert result.fuzzy_matched is True
        assert result.match_score >= 85

    def test_parse_fuzzy_modifier(self):
        """Fuzzy matching catches typos in modifiers."""
        result = parse_part_description("Sensor Housing - thredded")
        assert result.modifiers["threaded"] is True

    def test_parse_unknown_base_type(self):
        """Completely unknown base type returns UNKNOWN."""
        result = parse_part_description("Quantum Flux Capacitor - threaded")
        assert result.base_type == "UNKNOWN"
        assert result.fuzzy_matched is True
        assert result.match_score == 0.0
        # Known modifier should still be parsed
        assert result.modifiers["threaded"] is True

    def test_parse_unknown_modifier(self):
        """Unknown modifier is skipped; base type still parsed."""
        result = parse_part_description("Sensor Housing - cryogenic treated")
        assert result.base_type == "Sensor Housing"
        # Unknown modifier should not set any known flag
        assert result.complexity_score == 0

    def test_complexity_score_ordering(self):
        """Complex parts score higher than simple parts."""
        complex_result = parse_part_description("Manifold Block - complex internal channels")
        simple_result = parse_part_description("Mounting Bracket - standard")
        assert complex_result.complexity_score > simple_result.complexity_score

    def test_exact_match_score_is_100(self):
        """Exact matches report score of 100."""
        result = parse_part_description("Sensor Housing - threaded")
        assert result.match_score == 100.0

    def test_parse_all_real_data_descriptions(self, raw_data):
        """Every PartDescription in the actual CSV parses without error."""
        for desc in raw_data["PartDescription"].unique():
            result = parse_part_description(desc)
            assert result.base_type != "UNKNOWN", f"Failed to parse: {desc}"
            assert not result.fuzzy_matched, f"Required fuzzy match: {desc}"

    def test_parse_comma_separated_modifiers(self):
        """Parser handles comma-separated modifiers (future-proofing)."""
        result = parse_part_description("Sensor Housing - threaded, hardened")
        assert result.base_type == "Sensor Housing"
        assert result.modifiers["threaded"] is True
        assert result.modifiers["hardened"] is True
        assert result.complexity_score == 2  # threaded(1) + hardened(1)

    def test_parse_empty_string(self):
        """Empty string input doesn't crash."""
        result = parse_part_description("")
        assert result.base_type == "UNKNOWN"


class TestExtractDescriptionFeatures:
    """Tests for extract_description_features()."""

    def test_output_shape(self, sample_df):
        """Output has one row per input row."""
        features = extract_description_features(sample_df)
        assert len(features) == len(sample_df)

    def test_output_columns(self, sample_df):
        """Output has base_part_type, all flags, and complexity_score."""
        features = extract_description_features(sample_df)
        assert "base_part_type" in features.columns
        assert "complexity_score" in features.columns
        for flag in ALL_COMPLEXITY_FLAGS:
            assert flag in features.columns

    def test_no_nulls_in_output(self, sample_df):
        """No NaN values in extracted features."""
        features = extract_description_features(sample_df)
        assert not features.isna().any().any()

    def test_boolean_flags_are_bool(self, sample_df):
        """Complexity flags are boolean values."""
        features = extract_description_features(sample_df)
        for flag in ALL_COMPLEXITY_FLAGS:
            assert features[flag].dtype == bool


class TestBuildFeatureMatrix:
    """Tests for build_feature_matrix()."""

    def test_onehot_output_shape(self, sample_df):
        """One-hot feature matrix has expected dimensions."""
        X, y = build_feature_matrix(sample_df, encoding="onehot")
        assert len(X) == len(sample_df)
        assert len(y) == len(sample_df)
        # Should have many columns from one-hot encoding
        assert X.shape[1] > 10

    def test_label_output_shape(self, sample_df):
        """Label-encoded feature matrix has fewer columns than one-hot."""
        X_oh, _ = build_feature_matrix(sample_df, encoding="onehot")
        X_lbl, _ = build_feature_matrix(sample_df, encoding="label")
        assert X_lbl.shape[1] < X_oh.shape[1]

    def test_target_is_price(self, sample_df):
        """Target variable is TotalPrice_USD."""
        _, y = build_feature_matrix(sample_df, encoding="onehot")
        pd.testing.assert_series_equal(y, sample_df["TotalPrice_USD"])

    def test_log_quantity_values(self, sample_df):
        """log_quantity is correctly computed."""
        X, _ = build_feature_matrix(sample_df, encoding="onehot")
        expected = np.log(sample_df["Quantity"].astype(float))
        np.testing.assert_array_almost_equal(X["log_quantity"].values, expected.values)

    def test_log_quantity_no_nan_or_inf(self, raw_data):
        """log_quantity has no NaN or Inf values on real data."""
        X, _ = build_feature_matrix(raw_data, encoding="onehot")
        assert np.isfinite(X["log_quantity"]).all()

    def test_rush_job_binary(self, sample_df):
        """rush_job is 0 or 1."""
        X, _ = build_feature_matrix(sample_df, encoding="onehot")
        assert X["rush_job"].isin([0, 1]).all()

    def test_material_cost_tier_range(self, sample_df):
        """Material cost tier is in [1, 5] for non-null materials."""
        X, _ = build_feature_matrix(sample_df, encoding="onehot")
        valid = X["material_cost_tier"].dropna()
        assert valid.between(1, 5).all()

    def test_process_precision_tier_range(self, sample_df):
        """Process precision tier is in [1, 5] for non-null processes."""
        X, _ = build_feature_matrix(sample_df, encoding="onehot")
        valid = X["process_precision_tier"].dropna()
        assert valid.between(1, 5).all()

    def test_invalid_encoding_raises(self, sample_df):
        """ValueError raised for invalid encoding parameter."""
        with pytest.raises(ValueError, match="encoding must be"):
            build_feature_matrix(sample_df, encoding="invalid")

    def test_onehot_no_nulls_when_no_missing(self, sample_df):
        """No NaN in feature matrix when input has no missing values."""
        X, _ = build_feature_matrix(sample_df, encoding="onehot")
        # Tiers should be non-null when material/process are non-null
        cols = ["material_cost_tier", "process_precision_tier"]
        assert not X.drop(columns=cols).isna().any().any()

    def test_real_data_feature_matrix(self, raw_data):
        """Feature matrix builds from real data without error."""
        X, y = build_feature_matrix(raw_data, encoding="onehot")
        assert len(X) == 510
        assert len(y) == 510
        assert (y > 0).all()

    def test_real_data_label_encoding(self, raw_data):
        """Label-encoded feature matrix builds from real data without error."""
        X, y = build_feature_matrix(raw_data, encoding="label")
        assert len(X) == 510


class TestBuildFeatureMatrixNoEstimator:
    """Tests for build_feature_matrix_no_estimator()."""

    def test_no_estimator_columns(self, sample_df):
        """Output should not contain any estimator-related columns."""
        X, _ = build_feature_matrix_no_estimator(sample_df, encoding="onehot")
        estimator_cols = [c for c in X.columns if "estimator" in c.lower()]
        assert len(estimator_cols) == 0

    def test_fewer_columns_than_full(self, sample_df):
        """Should have fewer columns than the full feature matrix."""
        X_full, _ = build_feature_matrix(sample_df, encoding="onehot")
        X_no_est, _ = build_feature_matrix_no_estimator(sample_df, encoding="onehot")
        assert X_no_est.shape[1] < X_full.shape[1]

    def test_same_rows(self, sample_df):
        """Same number of rows as the full feature matrix."""
        X_full, _ = build_feature_matrix(sample_df, encoding="onehot")
        X_no_est, _ = build_feature_matrix_no_estimator(sample_df, encoding="onehot")
        assert X_no_est.shape[0] == X_full.shape[0]

    def test_label_encoding_no_estimator(self, sample_df):
        """Label-encoded version also excludes estimator."""
        X, _ = build_feature_matrix_no_estimator(sample_df, encoding="label")
        assert "estimator" not in X.columns


class TestFeatureMatrixWithMissing:
    """Tests for feature matrix behavior with missing values."""

    def test_missing_material_produces_nan_tier(self, sample_df):
        """Missing Material produces NaN in material_cost_tier."""
        sample_df.loc[sample_df.index[0], "Material"] = np.nan
        X, _ = build_feature_matrix(sample_df, encoding="onehot")
        assert np.isnan(X.loc[sample_df.index[0], "material_cost_tier"])

    def test_missing_process_produces_nan_tier(self, sample_df):
        """Missing Process produces NaN in process_precision_tier."""
        sample_df.loc[sample_df.index[0], "Process"] = np.nan
        X, _ = build_feature_matrix(sample_df, encoding="onehot")
        assert np.isnan(X.loc[sample_df.index[0], "process_precision_tier"])

    def test_real_data_nan_count_in_tiers(self, raw_data):
        """NaN count in tier columns matches missing count in source columns."""
        X, _ = build_feature_matrix(raw_data, encoding="onehot")
        assert X["material_cost_tier"].isna().sum() == raw_data["Material"].isna().sum()
        assert X["process_precision_tier"].isna().sum() == raw_data["Process"].isna().sum()
