"""Tests for the FastAPI quote endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from price_estimator.api import app, set_models

pytestmark = pytest.mark.api

V1 = "/v1"


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
    real overrides database.  Also reset the cached store."""
    db_file = tmp_path / "test_overrides.db"
    monkeypatch.setattr("price_estimator.api.DB_PATH", db_file)
    monkeypatch.setattr("price_estimator.api._store", None)


@pytest.fixture()
def client():
    """Return a TestClient with one mock model loaded."""
    set_models({"MockM6": MockModel()})
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /v1/quote
# ---------------------------------------------------------------------------


class TestCreateQuote:
    def test_success(self, client):
        resp = client.post(f"{V1}/quote", json=_valid_quote_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["estimate"] == 1000.0
        assert body["quote_id"].startswith("Q-API-")

    def test_no_models_returns_503(self, _isolate_db):
        set_models({})
        c = TestClient(app)
        resp = c.post(f"{V1}/quote", json=_valid_quote_payload())
        assert resp.status_code == 503
        body = resp.json()["detail"]["error"]
        assert body["code"] == "SERVICE_UNAVAILABLE"

    def test_missing_material_warning(self, client):
        resp = client.post(f"{V1}/quote", json=_valid_quote_payload(material=None))
        assert resp.status_code == 200
        warnings = resp.json()["warnings"]
        assert any("Missing material" in w for w in warnings)

    def test_missing_process_warning(self, client):
        resp = client.post(f"{V1}/quote", json=_valid_quote_payload(process=None))
        assert resp.status_code == 200
        warnings = resp.json()["warnings"]
        assert any("Missing process" in w for w in warnings)


# ---------------------------------------------------------------------------
# POST /v1/quote/{id}/override
# ---------------------------------------------------------------------------


class TestOverrideQuote:
    def _create_quote(self, client) -> str:
        resp = client.post(f"{V1}/quote", json=_valid_quote_payload())
        return resp.json()["quote_id"]

    def test_success(self, client):
        qid = self._create_quote(client)
        resp = client.post(
            f"{V1}/quote/{qid}/override",
            json={"human_price": 1500.0},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["stored"] is True
        assert body["delta_from_model"] == 500.0
        assert body["delta_pct"] == 50.0
        assert body["override_id"].startswith("OVR-")

    def test_override_stores_structured_reason(self, client):
        qid = self._create_quote(client)
        client.post(
            f"{V1}/quote/{qid}/override",
            json={
                "human_price": 1500.0,
                "reason_category": "material_hardness",
                "reason_text": "Inconel is harder than expected",
                "estimator_id": "Tanaka-san",
            },
        )
        resp = client.get(f"{V1}/quote/{qid}")
        body = resp.json()
        ovr = body["override"]
        assert ovr["reason_category"] == "material_hardness"
        assert ovr["reason_text"] == "Inconel is harder than expected"
        assert ovr["estimator_id"] == "Tanaka-san"

    def test_override_without_optional_fields(self, client):
        qid = self._create_quote(client)
        client.post(
            f"{V1}/quote/{qid}/override",
            json={"human_price": 1500.0},
        )
        resp = client.get(f"{V1}/quote/{qid}")
        body = resp.json()
        ovr = body["override"]
        assert ovr["reason_category"] is None
        assert ovr["reason_text"] is None
        assert ovr["estimator_id"] is None

    def test_not_found(self, client):
        resp = client.post(
            f"{V1}/quote/Q-NONEXISTENT/override",
            json={"human_price": 1500.0},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"]["code"] == "QUOTE_NOT_FOUND"


# ---------------------------------------------------------------------------
# GET /v1/quote/{id}
# ---------------------------------------------------------------------------


class TestGetQuote:
    def _create_quote(self, client) -> str:
        resp = client.post(f"{V1}/quote", json=_valid_quote_payload())
        return resp.json()["quote_id"]

    def test_no_override(self, client):
        qid = self._create_quote(client)
        resp = client.get(f"{V1}/quote/{qid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["original_estimate"] == 1000.0
        assert body["override"] is None
        assert body["final_price"] == 1000.0

    def test_with_override(self, client):
        qid = self._create_quote(client)
        client.post(
            f"{V1}/quote/{qid}/override",
            json={"human_price": 1500.0},
        )
        resp = client.get(f"{V1}/quote/{qid}")
        body = resp.json()
        assert body["final_price"] == 1500.0
        assert body["override"]["human_price"] == 1500.0

    def test_not_found(self, client):
        resp = client.get(f"{V1}/quote/Q-NONEXISTENT")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"]["code"] == "QUOTE_NOT_FOUND"


# ---------------------------------------------------------------------------
# Pydantic validation (422 — structural)
# ---------------------------------------------------------------------------


class TestStructuralValidation:
    def test_quantity_zero(self, client):
        resp = client.post(f"{V1}/quote", json=_valid_quote_payload(quantity=0))
        assert resp.status_code == 422
        body = resp.json()["error"]
        assert body["code"] == "UNPROCESSABLE_ENTITY"
        assert "fields" in body["details"]

    def test_quantity_negative(self, client):
        resp = client.post(f"{V1}/quote", json=_valid_quote_payload(quantity=-1))
        assert resp.status_code == 422

    def test_lead_time_zero(self, client):
        resp = client.post(
            f"{V1}/quote",
            json=_valid_quote_payload(lead_time_weeks=0),
        )
        assert resp.status_code == 422

    def test_lead_time_exceeds_max(self, client):
        resp = client.post(
            f"{V1}/quote",
            json=_valid_quote_payload(lead_time_weeks=53),
        )
        assert resp.status_code == 422

    def test_override_price_zero(self, client):
        resp = client.post(
            f"{V1}/quote/Q-FAKE/override",
            json={"human_price": 0},
        )
        assert resp.status_code == 422

    def test_override_price_exceeds_max(self, client):
        resp = client.post(
            f"{V1}/quote/Q-FAKE/override",
            json={"human_price": 10_000_001},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Business-rule validation (422 via Pydantic field_validator)
# ---------------------------------------------------------------------------


class TestBusinessRuleValidation:
    def test_unknown_material_rejected(self, client):
        resp = client.post(
            f"{V1}/quote",
            json=_valid_quote_payload(material="Unobtanium"),
        )
        assert resp.status_code == 422
        body = resp.json()["error"]
        assert any("material" in f["field"] for f in body["details"]["fields"])

    def test_unknown_process_rejected(self, client):
        resp = client.post(
            f"{V1}/quote",
            json=_valid_quote_payload(process="Laser Cutting"),
        )
        assert resp.status_code == 422

    def test_unknown_estimator_rejected(self, client):
        resp = client.post(
            f"{V1}/quote",
            json=_valid_quote_payload(estimator="Unknown-san"),
        )
        assert resp.status_code == 422

    def test_invalid_quantity_rejected(self, client):
        resp = client.post(
            f"{V1}/quote",
            json=_valid_quote_payload(quantity=7),
        )
        assert resp.status_code == 422

    def test_part_description_too_long(self, client):
        resp = client.post(
            f"{V1}/quote",
            json=_valid_quote_payload(part_description="A" * 201),
        )
        assert resp.status_code == 422

    def test_part_description_non_ascii(self, client):
        resp = client.post(
            f"{V1}/quote",
            json=_valid_quote_payload(part_description="Sensor Housing \U0001f600"),
        )
        assert resp.status_code == 422

    def test_part_description_blank_after_strip(self, client):
        resp = client.post(
            f"{V1}/quote",
            json=_valid_quote_payload(part_description="   "),
        )
        assert resp.status_code == 422

    def test_override_unknown_estimator_rejected(self, client):
        resp = client.post(
            f"{V1}/quote/Q-FAKE/override",
            json={"human_price": 1000.0, "estimator_id": "Unknown-san"},
        )
        assert resp.status_code == 422

    def test_override_reason_text_too_long(self, client):
        resp = client.post(
            f"{V1}/quote/Q-FAKE/override",
            json={"human_price": 1000.0, "reason_text": "x" * 1001},
        )
        assert resp.status_code == 422

    def test_null_material_allowed(self, client):
        """Null material is valid — it triggers a warning, not rejection."""
        resp = client.post(
            f"{V1}/quote",
            json=_valid_quote_payload(material=None),
        )
        assert resp.status_code == 200

    def test_unknown_part_description_allowed(self, client):
        """Unknown parts are allowed — fuzzy matching handles them."""
        resp = client.post(
            f"{V1}/quote",
            json=_valid_quote_payload(part_description="Flux Capacitor"),
        )
        assert resp.status_code == 200
