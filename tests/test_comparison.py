"""Tests for the 7-lens model comparison framework.

Includes:
- Unit tests for each lens function using sample data + simple models
- Boundary behavior validation suite (economic coherence, extrapolation, floors/ceilings)
- Integration tests against real data with trained models

The boundary tests serve double duty: they validate the comparison framework
itself AND act as a CI-runnable safety net for model quality (see VALIDATION_PLAN.md).
"""

import numpy as np
import pandas as pd
import pytest

from price_estimator.comparison import (
    _build_probe_row,
    _quantile_rank,
    boundary_behavior,
    calibration_bias,
    complexity_interpretability,
    economic_coherence,
    error_profile,
    format_scorecard_text,
    generate_scorecard,
    segment_fairness,
    stability_robustness,
)
from price_estimator.models import (
    M0LookupTable,
    M2RidgeLogLinear,
    XGBoostModel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_m0(raw_data):
    """M0 lookup table trained on real data."""
    model = M0LookupTable()
    model.fit(raw_data)
    return model


@pytest.fixture
def trained_m2(raw_data):
    """M2 log-linear Ridge trained on real data."""
    model = M2RidgeLogLinear()
    model.fit(raw_data)
    return model


@pytest.fixture
def trained_m6(raw_data):
    """M6 XGBoost trained on real data."""
    model = XGBoostModel(name="M6")
    model.fit(raw_data)
    return model


@pytest.fixture
def two_models(raw_data):
    """Dict of M0 and M2 trained on real data."""
    m0 = M0LookupTable()
    m0.fit(raw_data)
    m2 = M2RidgeLogLinear()
    m2.fit(raw_data)
    return {"M0": m0, "M2": m2}


# ---------------------------------------------------------------------------
# Probe row construction
# ---------------------------------------------------------------------------


class TestProbeRow:
    def test_returns_single_row_dataframe(self):
        probe = _build_probe_row()
        assert isinstance(probe, pd.DataFrame)
        assert len(probe) == 1

    def test_has_all_required_columns(self):
        probe = _build_probe_row()
        required = [
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
        for col in required:
            assert col in probe.columns

    def test_custom_values(self):
        probe = _build_probe_row(
            material="Inconel 718",
            quantity=50,
            rush=True,
        )
        assert probe.iloc[0]["Material"] == "Inconel 718"
        assert probe.iloc[0]["Quantity"] == 50
        assert probe.iloc[0]["RushJob"] == True  # noqa: E712 — numpy bool


# ---------------------------------------------------------------------------
# Quantile ranking
# ---------------------------------------------------------------------------


class TestQuantileRank:
    def test_higher_is_better(self):
        values = {"A": 90, "B": 70, "C": 50}
        ranks = _quantile_rank(values, higher_is_better=True)
        assert ranks["A"] >= ranks["B"] >= ranks["C"]

    def test_lower_is_better(self):
        values = {"A": 10, "B": 30, "C": 50}
        ranks = _quantile_rank(values, higher_is_better=False)
        assert ranks["A"] >= ranks["B"] >= ranks["C"]

    def test_ranks_are_1_to_5(self):
        values = {f"M{i}": float(i) for i in range(10)}
        ranks = _quantile_rank(values, higher_is_better=True)
        for r in ranks.values():
            assert 1 <= r <= 5

    def test_empty_input(self):
        assert _quantile_rank({}) == {}


# ---------------------------------------------------------------------------
# Lens 1: Error Profile
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestErrorProfile:
    def test_returns_all_models(self, raw_data, two_models):
        result = error_profile(two_models, raw_data)
        assert "M0" in result
        assert "M2" in result

    def test_has_required_keys(self, raw_data, two_models):
        result = error_profile(two_models, raw_data)
        for name, profile in result.items():
            assert "mape" in profile
            assert "median_ape" in profile
            assert "p90_ape" in profile
            assert "max_ape" in profile
            assert "pct_under_10" in profile
            assert "ape_values" in profile

    def test_percentiles_are_ordered(self, raw_data, two_models):
        result = error_profile(two_models, raw_data)
        for name, profile in result.items():
            assert profile["median_ape"] <= profile["p90_ape"]
            assert profile["p90_ape"] <= profile["p95_ape"]
            assert profile["p95_ape"] <= profile["max_ape"]

    def test_mape_is_positive(self, raw_data, two_models):
        result = error_profile(two_models, raw_data)
        for profile in result.values():
            assert profile["mape"] > 0


# ---------------------------------------------------------------------------
# Lens 2: Segment Fairness
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSegmentFairness:
    def test_returns_all_models(self, raw_data, two_models):
        result = segment_fairness(two_models, raw_data)
        assert set(result.keys()) == {"M0", "M2"}

    def test_has_expected_segments(self, raw_data, two_models):
        result = segment_fairness(two_models, raw_data)
        for name, segments in result.items():
            assert "by_material" in segments
            assert "by_part_type" in segments
            assert "by_quantity_band" in segments
            assert "by_estimator" in segments
            assert "by_price_quartile" in segments

    def test_segment_mapes_are_positive(self, raw_data, two_models):
        result = segment_fairness(two_models, raw_data)
        for segments in result.values():
            for seg_name, cats in segments.items():
                for cat, stats in cats.items():
                    assert stats["mape"] >= 0
                    assert stats["count"] > 0


# ---------------------------------------------------------------------------
# Lens 3: Economic Coherence
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEconomicCoherence:
    def test_returns_all_models(self, raw_data, two_models):
        result = economic_coherence(two_models, raw_data)
        assert set(result.keys()) == {"M0", "M2"}

    def test_has_checks(self, raw_data, two_models):
        result = economic_coherence(two_models, raw_data)
        for name, res in result.items():
            assert len(res["checks"]) > 0
            assert "pass_count" in res
            assert "total_count" in res
            assert "pass_rate" in res

    def test_check_names_are_known(self, raw_data, two_models):
        result = economic_coherence(two_models, raw_data)
        known_checks = {
            "material_ordering",
            "process_ordering",
            "quantity_discount",
            "rush_premium",
            "complexity_premium",
            "stochastic_material_ordering",
            "stochastic_rush_premium",
            "stochastic_quantity_discount",
            "stochastic_process_ordering",
            "stochastic_complexity_premium",
        }
        for res in result.values():
            for check in res["checks"]:
                assert check["name"] in known_checks

    def test_deterministic_checks_are_reproducible(self, raw_data, two_models):
        """Running twice should produce identical results."""
        r1 = economic_coherence(two_models, raw_data)
        r2 = economic_coherence(two_models, raw_data)
        for name in two_models:
            for c1, c2 in zip(r1[name]["checks"], r2[name]["checks"]):
                if c1["type"] == "deterministic":
                    assert c1["passed"] == c2["passed"]


# ---------------------------------------------------------------------------
# Lens 3: Economic Coherence — Boundary Validation Suite
# (These serve as CI-runnable domain invariant checks per VALIDATION_PLAN.md)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDomainInvariants:
    """Boundary validation suite: domain invariants that must hold.

    These tests verify the VALIDATION_PLAN.md Section 1 invariants
    against M0 (lookup table) and M2 (log-linear Ridge). M0 encodes
    economics directly, so it should pass all checks. M2 learns from
    data, so violations indicate model issues.
    """

    def test_m0_material_ordering(self, raw_data, trained_m0):
        """M0 must respect material cost ordering (domain invariant)."""
        materials = [
            "Aluminum 6061",
            "Aluminum 7075",
            "Stainless Steel 17-4 PH",
            "Titanium Grade 5",
            "Inconel 718",
        ]
        preds = []
        for mat in materials:
            probe = _build_probe_row(material=mat)
            preds.append(float(trained_m0.predict(probe)[0]))

        for i in range(len(preds) - 1):
            assert preds[i] < preds[i + 1], (
                f"Material ordering violated: {materials[i]} (${preds[i]:,.0f}) "
                f">= {materials[i + 1]} (${preds[i + 1]:,.0f})"
            )

    def test_m0_quantity_discount(self, raw_data, trained_m0):
        """M0 must produce decreasing unit prices as quantity increases."""
        quantities = [1, 5, 10, 20, 50, 100]
        unit_prices = []
        for qty in quantities:
            probe = _build_probe_row(quantity=qty)
            total = float(trained_m0.predict(probe)[0])
            unit_prices.append(total / qty)

        for i in range(len(unit_prices) - 1):
            assert unit_prices[i] >= unit_prices[i + 1], (
                f"Quantity discount violated: qty={quantities[i]} "
                f"(${unit_prices[i]:,.2f}/unit) < qty={quantities[i + 1]} "
                f"(${unit_prices[i + 1]:,.2f}/unit)"
            )

    def test_m0_rush_premium(self, raw_data, trained_m0):
        """M0 must charge more for rush jobs."""
        no_rush = float(trained_m0.predict(_build_probe_row(rush=False))[0])
        rush = float(trained_m0.predict(_build_probe_row(rush=True))[0])
        assert rush > no_rush, (
            f"Rush premium violated: rush=${rush:,.0f} <= no_rush=${no_rush:,.0f}"
        )

    def test_m2_rush_premium(self, raw_data, trained_m2):
        """M2 (learned from data) should also show rush premium."""
        no_rush = float(trained_m2.predict(_build_probe_row(rush=False))[0])
        rush = float(trained_m2.predict(_build_probe_row(rush=True))[0])
        assert rush > no_rush, (
            f"Rush premium violated: rush=${rush:,.0f} <= no_rush=${no_rush:,.0f}"
        )

    def test_m2_quantity_discount(self, raw_data, trained_m2):
        """M2 should produce decreasing unit prices with quantity."""
        quantities = [1, 5, 10, 20, 50, 100]
        unit_prices = []
        for qty in quantities:
            probe = _build_probe_row(quantity=qty)
            total = float(trained_m2.predict(probe)[0])
            unit_prices.append(total / qty)

        for i in range(len(unit_prices) - 1):
            assert unit_prices[i] >= unit_prices[i + 1], (
                f"Quantity discount violated at M2: qty={quantities[i]} "
                f"(${unit_prices[i]:,.2f}/unit) < qty={quantities[i + 1]} "
                f"(${unit_prices[i + 1]:,.2f}/unit)"
            )

    def test_m6_material_ordering(self, raw_data, trained_m6):
        """M6 must respect material cost ordering (domain invariant)."""
        materials = [
            "Aluminum 6061",
            "Aluminum 7075",
            "Stainless Steel 17-4 PH",
            "Titanium Grade 5",
            "Inconel 718",
        ]
        preds = []
        for mat in materials:
            probe = _build_probe_row(material=mat)
            preds.append(float(trained_m6.predict(probe)[0]))

        for i in range(len(preds) - 1):
            assert preds[i] < preds[i + 1], (
                f"Material ordering violated: {materials[i]} (${preds[i]:,.0f}) "
                f">= {materials[i + 1]} (${preds[i + 1]:,.0f})"
            )

    def test_m6_quantity_discount(self, raw_data, trained_m6):
        """M6 must produce decreasing unit prices at high quantities.

        XGBoost is a tree model so unit prices may not be strictly
        monotonic across all quantities. We verify the discount trend
        holds from qty=10 onwards where the model has seen enough
        training data to learn the discount curve.
        """
        quantities = [10, 20, 50, 100]
        unit_prices = []
        for qty in quantities:
            probe = _build_probe_row(quantity=qty)
            total = float(trained_m6.predict(probe)[0])
            unit_prices.append(total / qty)

        for i in range(len(unit_prices) - 1):
            assert unit_prices[i] >= unit_prices[i + 1], (
                f"Quantity discount violated at M6: qty={quantities[i]} "
                f"(${unit_prices[i]:,.2f}/unit) < qty={quantities[i + 1]} "
                f"(${unit_prices[i + 1]:,.2f}/unit)"
            )

    def test_m6_rush_premium(self, raw_data, trained_m6):
        """M6 (learned from data) should charge more for rush jobs."""
        no_rush = float(trained_m6.predict(_build_probe_row(rush=False))[0])
        rush = float(trained_m6.predict(_build_probe_row(rush=True))[0])
        assert rush > no_rush, (
            f"Rush premium violated: rush=${rush:,.0f} <= no_rush=${no_rush:,.0f}"
        )


# ---------------------------------------------------------------------------
# Lens 3: Boundary Safety Checks (VALIDATION_PLAN.md Section 3)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBoundarySafety:
    """Price floor, ceiling, and graceful degradation checks."""

    def test_m0_price_floor(self, raw_data, trained_m0):
        """Cheapest config should still predict >= $50."""
        probe = _build_probe_row(
            part_desc="Mounting Bracket - standard",
            material="Aluminum 6061",
            process="CNC Turning",
            quantity=1,
            rush=False,
        )
        pred = float(trained_m0.predict(probe)[0])
        assert pred >= 50, f"Price floor violated: ${pred:,.2f}"

    def test_m2_price_floor(self, raw_data, trained_m2):
        """Cheapest config should still predict >= $50."""
        probe = _build_probe_row(
            part_desc="Mounting Bracket - standard",
            material="Aluminum 6061",
            process="CNC Turning",
            quantity=1,
            rush=False,
        )
        pred = float(trained_m2.predict(probe)[0])
        assert pred >= 50, f"Price floor violated: ${pred:,.2f}"

    def test_m0_price_ceiling(self, raw_data, trained_m0):
        """Most expensive config should predict <= $500K."""
        probe = _build_probe_row(
            part_desc="Turbine Blade Housing - complex internal channels",
            material="Inconel 718",
            process="5-Axis Milling",
            quantity=100,
            rush=True,
        )
        pred = float(trained_m0.predict(probe)[0])
        assert pred <= 500_000, f"Price ceiling violated: ${pred:,.2f}"

    def test_m2_price_ceiling(self, raw_data, trained_m2):
        """Most expensive config should predict <= $500K."""
        probe = _build_probe_row(
            part_desc="Turbine Blade Housing - complex internal channels",
            material="Inconel 718",
            process="5-Axis Milling",
            quantity=100,
            rush=True,
        )
        pred = float(trained_m2.predict(probe)[0])
        assert pred <= 500_000, f"Price ceiling violated: ${pred:,.2f}"

    def test_m0_handles_missing_material(self, raw_data, trained_m0):
        """M0 should handle NaN material gracefully."""
        probe = _build_probe_row()
        probe.loc[0, "Material"] = np.nan
        pred = float(trained_m0.predict(probe)[0])
        assert pred > 0, "Prediction should be positive even with missing material"

    def test_m2_handles_missing_material(self, raw_data, trained_m2):
        """M2 should handle NaN material gracefully."""
        probe = _build_probe_row()
        probe.loc[0, "Material"] = np.nan
        pred = float(trained_m2.predict(probe)[0])
        assert pred > 0, "Prediction should be positive even with missing material"

    def test_m2_missing_material_reasonable_degradation(self, raw_data, trained_m2):
        """Prediction with missing material should be within 3x of with-material.

        Per VALIDATION_PLAN.md section 3.3, graceful degradation means the model
        should not produce wildly different predictions when a feature is missing.
        """
        probe_with = _build_probe_row(material="Aluminum 6061")
        pred_with = float(trained_m2.predict(probe_with)[0])

        probe_without = _build_probe_row(material="Aluminum 6061")
        probe_without.loc[0, "Material"] = np.nan
        pred_without = float(trained_m2.predict(probe_without)[0])

        ratio = max(pred_with, pred_without) / min(pred_with, pred_without)
        assert ratio <= 3.0, (
            f"Missing material degradation too large: "
            f"with=${pred_with:,.0f}, without=${pred_without:,.0f}, ratio={ratio:.2f}"
        )


# ---------------------------------------------------------------------------
# Lens 4: Calibration & Bias
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCalibrationBias:
    def test_returns_all_models(self, raw_data, two_models):
        result = calibration_bias(two_models, raw_data)
        assert set(result.keys()) == {"M0", "M2"}

    def test_has_required_keys(self, raw_data, two_models):
        result = calibration_bias(two_models, raw_data)
        for res in result.values():
            assert "mean_signed_error_pct" in res
            assert "pct_overestimated" in res
            assert "label" in res
            assert "by_price_quartile" in res

    def test_pct_overestimated_in_range(self, raw_data, two_models):
        result = calibration_bias(two_models, raw_data)
        for res in result.values():
            assert 0 <= res["pct_overestimated"] <= 100

    def test_label_is_valid(self, raw_data, two_models):
        result = calibration_bias(two_models, raw_data)
        valid_labels = {"conservative", "aggressive", "balanced"}
        for res in result.values():
            assert res["label"] in valid_labels


# ---------------------------------------------------------------------------
# Lens 5: Stability
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestStabilityRobustness:
    def test_returns_all_models(self, raw_data, two_models):
        result = stability_robustness(two_models, raw_data, n_bootstrap=5)
        assert set(result.keys()) == {"M0", "M2"}

    def test_has_fold_mapes(self, raw_data, two_models):
        result = stability_robustness(two_models, raw_data, n_bootstrap=5)
        for res in result.values():
            assert len(res["fold_mapes"]) == 5
            assert res["fold_mape_std"] >= 0

    def test_bootstrap_cv_is_nonnegative(self, raw_data, two_models):
        result = stability_robustness(two_models, raw_data, n_bootstrap=5)
        for res in result.values():
            assert res["bootstrap_mean_cv_pct"] >= 0


# ---------------------------------------------------------------------------
# Lens 6: Boundary Behavior
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBoundaryBehavior:
    def test_returns_all_models(self, raw_data, two_models):
        result = boundary_behavior(two_models, raw_data)
        assert set(result.keys()) == {"M0", "M2"}

    def test_has_tests(self, raw_data, two_models):
        result = boundary_behavior(two_models, raw_data)
        for res in result.values():
            assert len(res["tests"]) > 0
            assert "pass_count" in res


# ---------------------------------------------------------------------------
# Lens 7: Complexity
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestComplexityInterpretability:
    def test_returns_all_models(self, raw_data, two_models):
        result = complexity_interpretability(two_models, raw_data)
        assert set(result.keys()) == {"M0", "M2"}

    def test_has_required_keys(self, raw_data, two_models):
        result = complexity_interpretability(two_models, raw_data)
        for res in result.values():
            assert "train_seconds" in res
            assert "effective_params" in res
            assert "interpretability_rating" in res

    def test_m0_is_high_interpretability(self, raw_data, two_models):
        result = complexity_interpretability(two_models, raw_data)
        assert result["M0"]["interpretability_rating"] == "high"

    def test_m2_is_high_interpretability(self, raw_data, two_models):
        result = complexity_interpretability(two_models, raw_data)
        assert result["M2"]["interpretability_rating"] == "high"


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestScorecard:
    def test_generates_scorecard(self, raw_data, two_models):
        err = error_profile(two_models, raw_data)
        seg = segment_fairness(two_models, raw_data)
        coh = economic_coherence(two_models, raw_data)
        cal = calibration_bias(two_models, raw_data)
        stab = stability_robustness(two_models, raw_data, n_bootstrap=5)
        bound = boundary_behavior(two_models, raw_data)
        comp = complexity_interpretability(two_models, raw_data)

        scorecard = generate_scorecard(err, seg, coh, cal, stab, bound, comp)
        assert "M0" in scorecard
        assert "M2" in scorecard

        for name, sc in scorecard.items():
            assert "ranks" in sc
            assert "dots" in sc
            assert "average_rank" in sc
            assert "narrative" in sc
            assert 1.0 <= sc["average_rank"] <= 5.0

    def test_format_text_is_nonempty(self, raw_data, two_models):
        err = error_profile(two_models, raw_data)
        seg = segment_fairness(two_models, raw_data)
        coh = economic_coherence(two_models, raw_data)
        cal = calibration_bias(two_models, raw_data)
        stab = stability_robustness(two_models, raw_data, n_bootstrap=5)
        bound = boundary_behavior(two_models, raw_data)
        comp = complexity_interpretability(two_models, raw_data)

        scorecard = generate_scorecard(err, seg, coh, cal, stab, bound, comp)
        text = format_scorecard_text(scorecard)
        assert len(text) > 100
        assert "MODEL COMPARISON REPORT" in text
        assert "NARRATIVES" in text
