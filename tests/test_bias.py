"""Tests for the estimator bias analysis module."""

import numpy as np
import pytest

from price_estimator.bias import _bootstrap_ci, compute_estimator_bias, format_bias_report

# ---------------------------------------------------------------------------
# _bootstrap_ci tests (fast)
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Tests for the _bootstrap_ci helper."""

    def test_constant_values(self):
        """Constant array should produce CI very close to that constant."""
        values = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        low, high = _bootstrap_ci(values)
        assert low == pytest.approx(10.0, abs=0.01)
        assert high == pytest.approx(10.0, abs=0.01)

    def test_ci_ordering(self):
        """Lower bound must be <= upper bound."""
        rng = np.random.default_rng(99)
        values = rng.normal(loc=5.0, scale=2.0, size=50)
        low, high = _bootstrap_ci(values)
        assert low <= high

    def test_single_value(self):
        """Single-element array should not crash."""
        values = np.array([42.0])
        low, high = _bootstrap_ci(values)
        assert low == pytest.approx(42.0, abs=0.01)
        assert high == pytest.approx(42.0, abs=0.01)

    def test_ci_contains_mean(self):
        """The sample mean should fall within the CI."""
        rng = np.random.default_rng(7)
        values = rng.normal(loc=0.0, scale=1.0, size=200)
        low, high = _bootstrap_ci(values)
        assert low <= np.mean(values) <= high


# ---------------------------------------------------------------------------
# format_bias_report tests (fast)
# ---------------------------------------------------------------------------


class TestFormatBiasReport:
    """Tests for the format_bias_report function."""

    def test_report_contains_labels_and_names(self):
        """Report string should include estimator names and labels."""
        bias_results = {
            "summary": {
                "Sato-san": {
                    "mean_pct_bias": 5.2,
                    "median_pct_bias": 4.8,
                    "ci_95_low": 2.1,
                    "ci_95_high": 8.3,
                    "std_pct_bias": 3.1,
                    "n_quotes": 170,
                    "label": "safe",
                },
                "Tanaka-san": {
                    "mean_pct_bias": -3.1,
                    "median_pct_bias": -2.9,
                    "ci_95_low": -5.0,
                    "ci_95_high": -1.2,
                    "std_pct_bias": 2.5,
                    "n_quotes": 170,
                    "label": "aggressive",
                },
                "Suzuki-san": {
                    "mean_pct_bias": 0.3,
                    "median_pct_bias": 0.1,
                    "ci_95_low": -1.0,
                    "ci_95_high": 1.6,
                    "std_pct_bias": 2.0,
                    "n_quotes": 170,
                    "label": "neutral",
                },
            },
            "by_part_type": {
                "Sato-san": {
                    "housing": {"mean_pct_bias": 6.0, "n_quotes": 30},
                },
                "Tanaka-san": {
                    "bracket": {"mean_pct_bias": -2.0, "n_quotes": 25},
                },
                "Suzuki-san": {
                    "manifold": {"mean_pct_bias": 0.5, "n_quotes": 20},
                },
            },
            "by_material": {},
            "over_time": {},
        }
        report = format_bias_report(bias_results)
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Sato-san" in report
        assert "Tanaka-san" in report
        assert "Suzuki-san" in report
        assert "SAFE" in report
        assert "AGGRESSIVE" in report
        assert "NEUTRAL" in report


# ---------------------------------------------------------------------------
# compute_estimator_bias smoke test (sample_df — slow, trains XGBoost)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestComputeEstimatorBiasSample:
    """Smoke tests using the small sample_df fixture."""

    def test_returns_expected_keys(self, sample_df):
        """Result dict must contain all four top-level keys."""
        result = compute_estimator_bias(sample_df)
        assert set(result.keys()) == {
            "summary",
            "by_part_type",
            "by_material",
            "over_time",
        }

    def test_all_estimators_present(self, sample_df):
        """All estimators in the input should appear in summary."""
        result = compute_estimator_bias(sample_df)
        expected = set(sample_df["Estimator"].unique())
        assert set(result["summary"].keys()) == expected

    def test_summary_entry_keys(self, sample_df):
        """Each estimator summary must have the required fields."""
        result = compute_estimator_bias(sample_df)
        required_keys = {
            "mean_pct_bias",
            "median_pct_bias",
            "ci_95_low",
            "ci_95_high",
            "std_pct_bias",
            "n_quotes",
            "label",
        }
        for est, stats in result["summary"].items():
            assert set(stats.keys()) == required_keys, f"Missing keys for {est}"

    def test_labels_are_valid(self, sample_df):
        """Labels must be one of the three allowed values."""
        result = compute_estimator_bias(sample_df)
        valid_labels = {"safe", "aggressive", "neutral"}
        for est, stats in result["summary"].items():
            assert stats["label"] in valid_labels, f"Invalid label '{stats['label']}' for {est}"


# ---------------------------------------------------------------------------
# Integration test (raw_data — slow, trains XGBoost on full dataset)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
class TestComputeEstimatorBiasIntegration:
    """Integration tests on the real 510-row dataset."""

    def test_all_estimators_present(self, raw_data):
        """All 3 estimators should appear in summary."""
        result = compute_estimator_bias(raw_data)
        assert len(result["summary"]) == 3

    def test_n_quotes_sum(self, raw_data):
        """Total n_quotes across estimators should equal 510."""
        result = compute_estimator_bias(raw_data)
        total = sum(stats["n_quotes"] for stats in result["summary"].values())
        assert total == 510

    def test_ci_reasonable_width(self, raw_data):
        """95% CIs should not be absurdly wide (< 50pp spread)."""
        result = compute_estimator_bias(raw_data)
        for est, stats in result["summary"].items():
            ci_width = stats["ci_95_high"] - stats["ci_95_low"]
            assert ci_width < 50, f"CI width {ci_width:.1f}% too wide for {est}"
            assert ci_width >= 0, "CI low should be <= CI high"
