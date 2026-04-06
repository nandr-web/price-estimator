"""Tests for the FastAPI quote endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from price_estimator.api import app, set_models

pytestmark = pytest.mark.api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockModel:
    """Deterministic model that always predicts 1000.0."""

    name = "MockM6"

    def predict(self, df):
        return np.array([1000.0] * len(df))

    def fit(self, df):
        return self


def _valid_quote_payload(**overrides) -> dict:
    base = {
        "part_description": "Sensor Housing - threaded",
        "material": "Inconel 718",
        "process": "Wire EDM",
        "quantity": 5,
        "rush_job": False,
        "lead_time_weeks": 4,
        "estimator": "Sato-san",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point the API's DB_PATH at a temp file so tests never touch the
    real overrides database."""
    db_file = tmp_path / "test_overrides.db"
    monkeypatch.setattr("price_estimator.api.DB_PATH", db_file)


@pytest.fixture()
def client():
    """Return a TestClient with one mock model loaded."""
    set_models({"MockM6": MockModel()})
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /quote
# ---------------------------------------------------------------------------


class TestCreateQuote:
    def test_success(self, client):
        resp = client.post("/quote", json=_valid_quote_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["estimate"] == 1000.0
        assert body["quote_id"].startswith("Q-API-")
        assert body["model_range"]["low"] == 1000.0
        assert body["model_range"]["high"] == 1000.0

    def test_no_models_returns_503(self, _isolate_db):
        set_models({})
        c = TestClient(app)
        resp = c.post("/quote", json=_valid_quote_payload())
        assert resp.status_code == 503

    def test_missing_material_flag(self, client):
        resp = client.post("/quote", json=_valid_quote_payload(material=None))
        assert resp.status_code == 200
        flags = resp.json()["confidence_flags"]
        assert any("Missing material" in f for f in flags)

    def test_missing_process_flag(self, client):
        resp = client.post("/quote", json=_valid_quote_payload(process=None))
        assert resp.status_code == 200
        flags = resp.json()["confidence_flags"]
        assert any("Missing process" in f for f in flags)


# ---------------------------------------------------------------------------
# POST /quote/{id}/override
# ---------------------------------------------------------------------------


class TestOverrideQuote:
    def _create_quote(self, client) -> str:
        resp = client.post("/quote", json=_valid_quote_payload())
        return resp.json()["quote_id"]

    def test_success(self, client):
        qid = self._create_quote(client)
        resp = client.post(
            f"/quote/{qid}/override",
            json={"human_price": 1500.0},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["stored"] is True
        assert body["delta_from_model"] == 500.0
        assert body["delta_pct"] == 50.0
        assert body["override_id"].startswith("OVR-")

    def test_not_found(self, client):
        resp = client.post(
            "/quote/Q-NONEXISTENT/override",
            json={"human_price": 1500.0},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /quote/{id}
# ---------------------------------------------------------------------------


class TestGetQuote:
    def _create_quote(self, client) -> str:
        resp = client.post("/quote", json=_valid_quote_payload())
        return resp.json()["quote_id"]

    def test_no_override(self, client):
        qid = self._create_quote(client)
        resp = client.get(f"/quote/{qid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["original_estimate"] == 1000.0
        assert body["human_override"] is None
        assert body["final_price"] == 1000.0

    def test_with_override(self, client):
        qid = self._create_quote(client)
        client.post(
            f"/quote/{qid}/override",
            json={"human_price": 1500.0},
        )
        resp = client.get(f"/quote/{qid}")
        body = resp.json()
        assert body["final_price"] == 1500.0
        assert body["human_override"]["human_price"] == 1500.0

    def test_not_found(self, client):
        resp = client.get("/quote/Q-NONEXISTENT")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_quantity_zero(self, client):
        resp = client.post("/quote", json=_valid_quote_payload(quantity=0))
        assert resp.status_code == 422

    def test_quantity_negative(self, client):
        resp = client.post("/quote", json=_valid_quote_payload(quantity=-1))
        assert resp.status_code == 422

    def test_lead_time_zero(self, client):
        resp = client.post("/quote", json=_valid_quote_payload(lead_time_weeks=0))
        assert resp.status_code == 422

    def test_lead_time_exceeds_max(self, client):
        resp = client.post("/quote", json=_valid_quote_payload(lead_time_weeks=53))
        assert resp.status_code == 422
