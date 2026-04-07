"""Smoke tests, integration tests, and factory tests for all 14 models."""

import joblib
import numpy as np
import pytest

from price_estimator.models import (
    CVResults,
    M2RidgeLogLinear,
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


@pytest.mark.slow
def test_m2_jensens_correction_applied(raw_data):
    """M2 Jensen's correction must increase predictions above raw exp(log_pred).

    Trains M2, verifies _residual_var > 0, then checks that the final
    prediction is higher than exp(raw_log_pred) for a probe row.
    """
    from price_estimator.features import build_feature_matrix

    m2 = M2RidgeLogLinear()
    m2.fit(raw_data)

    assert m2._residual_var > 0, "Residual variance should be positive after training"

    # Build a single probe row using the same feature construction M2 uses
    probe = raw_data.iloc[[0]].copy()
    X, _ = build_feature_matrix(probe, encoding="onehot")
    X = X.fillna(0)
    X = X.reindex(columns=m2._feature_cols, fill_value=0)

    raw_log_pred = float(m2.pipeline.predict(X.values)[0])
    naive_pred = np.exp(raw_log_pred)
    corrected_pred = float(m2.predict(probe)[0])

    assert corrected_pred > naive_pred, (
        f"Jensen's correction should increase prediction: "
        f"corrected={corrected_pred:.2f} <= naive={naive_pred:.2f}"
    )


@pytest.mark.slow
def test_m2_beats_m1_on_real_data(raw_data):
    """M2 (log-linear) should beat M1 (additive) by at least 50pp MAPE.

    M1 is ~93% MAPE, M2 is ~10.8% MAPE. The log transform is critical.
    """
    m1 = get_model_by_name("M1")
    m2 = get_model_by_name("M2")

    cv_m1 = m1.cross_validate(raw_data)
    cv_m2 = m2.cross_validate(raw_data)

    m1_mape = cv_m1.mean_metrics["MAPE"]
    m2_mape = cv_m2.mean_metrics["MAPE"]

    assert m2_mape < m1_mape - 50, (
        f"M2 should beat M1 by at least 50pp MAPE: "
        f"M1={m1_mape:.1f}%, M2={m2_mape:.1f}%, diff={m1_mape - m2_mape:.1f}pp"
    )


@pytest.mark.slow
def test_model_serialization_roundtrip(raw_data, tmp_path):
    """M6 serialized and reloaded must produce identical predictions."""
    m6 = get_model_by_name("M6")
    m6.fit(raw_data)
    preds_before = m6.predict(raw_data)

    model_path = tmp_path / "M6.joblib"
    joblib.dump(m6, model_path)
    m6_reloaded = joblib.load(model_path)
    preds_after = m6_reloaded.predict(raw_data)

    np.testing.assert_array_equal(
        preds_before,
        preds_after,
        err_msg="Predictions differ after serialization roundtrip",
    )
