"""Tests for prediction utilities: OOD detection, model disagreement, SHAP formatting."""

import numpy as np
import pandas as pd
import pytest

from price_estimator.predict import (
    TrainingBounds,
    compute_model_disagreement,
    detect_ood,
    format_shap_explanation,
)


class TestTrainingBounds:
    """Tests for TrainingBounds.from_dataframe() and JSON serialization."""

    def test_from_dataframe_quantity_range(self, sample_df):
        """Quantity range matches min/max of sample data."""
        bounds = TrainingBounds.from_dataframe(sample_df)
        assert bounds.quantity_range == (1, 50)

    def test_from_dataframe_known_materials(self, sample_df):
        """Known materials set matches unique materials in data."""
        bounds = TrainingBounds.from_dataframe(sample_df)
        expected = {
            "Inconel 718",
            "Titanium Grade 5",
            "Aluminum 6061",
            "Aluminum 7075",
            "Stainless Steel 17-4 PH",
        }
        assert bounds.known_materials == expected

    def test_from_dataframe_handles_missing(self, sample_df):
        """NaN materials are excluded from the known set."""
        df = sample_df.copy()
        df.loc[0, "Material"] = np.nan
        bounds = TrainingBounds.from_dataframe(df)
        assert np.nan not in bounds.known_materials
        assert len(bounds.known_materials) == 4

    def test_json_round_trip(self, sample_df):
        """to_json → from_json preserves all fields exactly."""
        original = TrainingBounds.from_dataframe(sample_df)
        restored = TrainingBounds.from_json(original.to_json())
        assert restored.quantity_range == original.quantity_range
        assert restored.lead_time_range == original.lead_time_range
        assert restored.known_materials == original.known_materials
        assert restored.known_processes == original.known_processes
        assert restored.known_part_types == original.known_part_types
        assert restored.known_estimators == original.known_estimators

    def test_to_json_types(self, sample_df):
        """to_json produces JSON-serializable types (lists, not sets/tuples)."""
        import json

        bounds = TrainingBounds.from_dataframe(sample_df)
        data = bounds.to_json()
        # Should not raise — all values are JSON-serializable
        json.dumps(data)
        assert isinstance(data["quantity_range"], list)
        assert isinstance(data["known_materials"], list)


class TestDetectOOD:
    """Tests for out-of-distribution detection."""

    @pytest.fixture
    def bounds(self, sample_df):
        """Training bounds from sample data."""
        return TrainingBounds.from_dataframe(sample_df)

    def _make_row(self, **overrides):
        """Build a single-row DataFrame with valid defaults, applying overrides."""
        defaults = {
            "Quantity": 10,
            "LeadTimeWeeks": 5,
            "Material": "Aluminum 6061",
            "Process": "CNC Turning",
            "PartDescription": "Mounting Bracket - standard",
            "Estimator": "Sato-san",
        }
        defaults.update(overrides)
        return pd.DataFrame([defaults])

    def test_in_distribution_no_flags(self, bounds):
        """A valid row within bounds is not flagged."""
        row = self._make_row()
        results = detect_ood(row, bounds)
        assert len(results) == 1
        assert results[0]["is_ood"] is False
        assert results[0]["reasons"] == []

    def test_quantity_out_of_range(self, bounds):
        """Quantity above training max is flagged."""
        row = self._make_row(Quantity=200)
        results = detect_ood(row, bounds)
        assert results[0]["is_ood"] is True
        assert any("Quantity" in r for r in results[0]["reasons"])

    def test_lead_time_out_of_range(self, bounds):
        """Lead time above training max is flagged."""
        row = self._make_row(LeadTimeWeeks=20)
        results = detect_ood(row, bounds)
        assert results[0]["is_ood"] is True
        assert any("LeadTimeWeeks" in r for r in results[0]["reasons"])

    def test_unknown_material(self, bounds):
        """Unknown material is flagged."""
        row = self._make_row(Material="Unobtanium")
        results = detect_ood(row, bounds)
        assert results[0]["is_ood"] is True
        assert any("Unobtanium" in r for r in results[0]["reasons"])

    def test_missing_material(self, bounds):
        """NaN material is flagged with 'Missing material'."""
        row = self._make_row(Material=np.nan)
        results = detect_ood(row, bounds)
        assert results[0]["is_ood"] is True
        assert any("Missing material" in r for r in results[0]["reasons"])

    def test_missing_process(self, bounds):
        """NaN process is flagged."""
        row = self._make_row(Process=np.nan)
        results = detect_ood(row, bounds)
        assert results[0]["is_ood"] is True
        assert any("Missing process" in r for r in results[0]["reasons"])

    def test_unknown_part_description(self, bounds):
        """Unknown part description is flagged."""
        row = self._make_row(PartDescription="Widget - fancy")
        results = detect_ood(row, bounds)
        assert results[0]["is_ood"] is True
        assert any("Widget" in r for r in results[0]["reasons"])

    def test_multiple_reasons(self, bounds):
        """Row with multiple OOD conditions lists all reasons."""
        row = self._make_row(Material="Unobtanium", Quantity=200)
        results = detect_ood(row, bounds)
        assert results[0]["is_ood"] is True
        reasons = results[0]["reasons"]
        assert len(reasons) >= 2
        assert any("Quantity" in r for r in reasons)
        assert any("Unobtanium" in r for r in reasons)


class TestComputeModelDisagreement:
    """Tests for multi-model disagreement statistics."""

    def test_single_model(self):
        """Single model produces zero spread and no flagged indices."""
        preds = {"M1": np.array([100.0, 200.0, 300.0])}
        result = compute_model_disagreement(preds)
        assert np.all(result["spread_pct"] == pytest.approx(0.0))
        assert len(result["flagged_indices"]) == 0

    def test_two_agreeing_models(self):
        """Two models with similar predictions produce low spread."""
        preds = {
            "M1": np.array([100.0, 200.0, 300.0]),
            "M2": np.array([101.0, 202.0, 303.0]),
        }
        result = compute_model_disagreement(preds)
        # Spread should be well under 20% for all rows
        assert np.all(result["spread_pct"] < 20.0)
        assert len(result["flagged_indices"]) == 0

    def test_two_disagreeing_models(self):
        """Models differing by >20% produce flagged indices."""
        preds = {
            "M1": np.array([100.0, 200.0]),
            "M2": np.array([150.0, 300.0]),
        }
        result = compute_model_disagreement(preds)
        assert len(result["flagged_indices"]) > 0

    def test_median_is_correct(self):
        """Median across 3 models is computed correctly."""
        preds = {
            "M1": np.array([100.0, 200.0]),
            "M2": np.array([200.0, 400.0]),
            "M3": np.array([300.0, 600.0]),
        }
        result = compute_model_disagreement(preds)
        expected_median = np.array([200.0, 400.0])
        assert result["median_pred"] == pytest.approx(expected_median)


class TestFormatShapExplanation:
    """Tests for SHAP explanation formatting."""

    def _make_explanation(self, n_features=4):
        """Build a synthetic SHAP explanation dict."""
        return {
            "shap_values": np.array([10.0, -5.0, 3.0, -1.0][:n_features]),
            "feature_names": ["Quantity", "Material", "Process", "LeadTime"][:n_features],
            "base_value": 1000.0,
        }

    def test_format_basic(self):
        """Output contains base value, feature names, and prediction."""
        explanation = self._make_explanation()
        output = format_shap_explanation(explanation, top_n=5)
        assert "Base value" in output
        assert "Quantity" in output
        assert "Material" in output
        assert "Prediction" in output

    def test_top_n_limits_features(self):
        """Only top_n features appear between Base value and Prediction lines."""
        explanation = self._make_explanation(n_features=4)
        output = format_shap_explanation(explanation, top_n=2)
        lines = output.strip().split("\n")
        # First line is "Base value: ...", last is "Prediction: ..."
        # Feature lines are in between
        feature_lines = [line for line in lines if line.startswith("  ")]
        assert len(feature_lines) == 2
