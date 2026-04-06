"""Smoke tests, integration tests, and factory tests for all 14 models."""

import numpy as np
import pytest

from price_estimator.models import (
    CVResults,
    compute_metrics,
    get_all_models,
    get_model_by_name,
)

# ---------------------------------------------------------------------------
# Model names for parametrized smoke tests
# ---------------------------------------------------------------------------

ALL_MODEL_NAMES = [m.name for m in get_all_models()]


# ---------------------------------------------------------------------------
# Smoke tests — fast, use sample_df (5 rows)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
def test_smoke_fit_predict(sample_df, model_name):
    """Every model can fit and predict on the 5-row sample without error."""
    model = get_model_by_name(model_name)
    model.fit(sample_df)
    preds = model.predict(sample_df)

    assert isinstance(preds, np.ndarray), "predictions must be a numpy array"
    assert len(preds) == len(sample_df), "prediction length must match input"
    assert np.all(np.isfinite(preds)), f"non-finite values in {model_name} output"
    assert np.all(preds > 0), f"non-positive values in {model_name} output"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility_m6(sample_df):
    """M6 trained twice on the same data produces identical predictions."""
    m1 = get_model_by_name("M6")
    m1.fit(sample_df)
    p1 = m1.predict(sample_df)

    m2 = get_model_by_name("M6")
    m2.fit(sample_df)
    p2 = m2.predict(sample_df)

    np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_get_all_models_count():
    """Factory returns exactly 14 models."""
    models = get_all_models()
    assert len(models) == 14


def test_get_all_models_unique_names():
    """All model names are unique."""
    names = [m.name for m in get_all_models()]
    assert len(names) == len(set(names))


def test_get_model_by_name_valid():
    """get_model_by_name returns the correct model for each known name."""
    for name in ALL_MODEL_NAMES:
        model = get_model_by_name(name)
        assert model.name == name


def test_get_model_by_name_invalid():
    """get_model_by_name raises ValueError for unknown names."""
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_by_name("INVALID")


# ---------------------------------------------------------------------------
# compute_metrics unit test
# ---------------------------------------------------------------------------


def test_compute_metrics_known_values():
    """compute_metrics returns correct MAPE, MedAPE, RMSE, R2."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])

    m = compute_metrics(y_true, y_pred)

    # APE = |10/100|, |10/200|, |30/300| = 0.1, 0.05, 0.1
    expected_mape = float(np.mean([0.1, 0.05, 0.1]) * 100)
    assert abs(m["MAPE"] - expected_mape) < 1e-8

    expected_med_ape = float(np.median([0.1, 0.05, 0.1]) * 100)
    assert abs(m["MedAPE"] - expected_med_ape) < 1e-8

    expected_rmse = float(np.sqrt(np.mean([10.0**2, 10.0**2, 30.0**2])))
    assert abs(m["RMSE"] - expected_rmse) < 1e-8

    assert -10 < m["R2"] <= 1.0  # sanity bound


# ---------------------------------------------------------------------------
# Integration tests — use real CSV, marked slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cv_m0_integration(raw_data):
    """M0 cross-validation on the full dataset produces valid results."""
    model = get_model_by_name("M0")
    results = model.cross_validate(raw_data)

    assert isinstance(results, CVResults)
    assert len(results.fold_metrics) == 5
    assert results.mean_metrics["MAPE"] > 0
    assert np.isfinite(results.mean_metrics["MAPE"])
    assert len(results.all_predictions) == len(raw_data)


@pytest.mark.slow
def test_cv_m6_integration(raw_data):
    """M6 cross-validation on the full dataset produces valid results."""
    model = get_model_by_name("M6")
    results = model.cross_validate(raw_data)

    assert isinstance(results, CVResults)
    assert len(results.fold_metrics) == 5
    assert results.mean_metrics["MAPE"] > 0
    assert np.isfinite(results.mean_metrics["MAPE"])
    assert len(results.all_predictions) == len(raw_data)
