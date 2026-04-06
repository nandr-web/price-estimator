"""FastAPI application for the human-in-the-loop quote workflow.

Endpoints:
    POST /quote         - Get an AI price estimate with bands and SHAP explanation
    POST /quote/{id}/override - Override an AI quote with human price
    GET  /quote/{id}    - Retrieve a quote (original + override if any)
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticModel
from pydantic import Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Price Estimator API",
    description="AI-assisted quoting for precision machined aerospace parts",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = Path("outputs/overrides.db")


def get_db() -> sqlite3.Connection:
    """Get a SQLite connection, creating tables if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quotes (
            quote_id TEXT PRIMARY KEY,
            features TEXT NOT NULL,
            model_price REAL NOT NULL,
            model_range_low REAL,
            model_range_high REAL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS overrides (
            override_id TEXT PRIMARY KEY,
            quote_id TEXT NOT NULL,
            human_price REAL NOT NULL,
            reason_category TEXT,
            reason_text TEXT,
            estimator_id TEXT,
            delta_from_model REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (quote_id) REFERENCES quotes(quote_id)
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class OverrideReasonCategory(StrEnum):
    """Structured override reason categories."""

    MATERIAL_HARDNESS = "material_hardness"
    GEOMETRY_COMPLEXITY = "geometry_complexity"
    SURFACE_FINISH = "surface_finish"
    TOOLING_DIFFICULTY = "tooling_difficulty"
    CUSTOMER_RELATIONSHIP = "customer_relationship"
    SCRAP_RISK = "scrap_risk"
    CERTIFICATION_REQUIREMENTS = "certification_requirements"
    OTHER = "other"


class QuoteRequest(PydanticModel):
    """Input for a new quote request."""

    part_description: str
    material: str | None = None
    process: str | None = None
    quantity: int = Field(gt=0)
    rush_job: bool = False
    lead_time_weeks: int = Field(ge=1, le=52)
    estimator: str | None = None


class QuoteResponse(PydanticModel):
    """Output from a quote request."""

    quote_id: str
    estimate: float
    model_range: dict | None = None
    prediction_interval: dict | None = None
    estimator_range: dict | None = None
    confidence_flags: list[str] = []
    shap_explanation: list[dict] | None = None


class OverrideRequest(PydanticModel):
    """Input for overriding a quote."""

    human_price: float = Field(gt=0)
    reason_category: OverrideReasonCategory | None = None
    reason_text: str | None = None
    estimator_id: str | None = None


class OverrideResponse(PydanticModel):
    """Output from an override request."""

    stored: bool
    override_id: str
    delta_from_model: float
    delta_pct: float


class QuoteDetail(PydanticModel):
    """Full quote detail with optional override."""

    quote_id: str
    original_estimate: float
    human_override: dict | None = None
    final_price: float


# ---------------------------------------------------------------------------
# State (loaded at startup)
# ---------------------------------------------------------------------------

# These are set by the serve script after loading models
_models: dict = {}
_training_bounds = None


def set_models(models: dict, training_bounds=None):
    """Set the loaded models for the API to use.

    Called by the serve script after loading serialized models.

    Args:
        models: Dict mapping model name to trained model instance.
        training_bounds: TrainingBounds instance for OOD detection.
    """
    global _models, _training_bounds
    _models = models
    _training_bounds = training_bounds


def _request_to_dataframe(req: QuoteRequest) -> pd.DataFrame:
    """Convert a QuoteRequest to a single-row DataFrame."""
    return pd.DataFrame(
        [
            {
                "QuoteID": "API",
                "Date": pd.Timestamp.now(),
                "PartDescription": req.part_description,
                "Material": req.material,
                "Process": req.process,
                "Quantity": req.quantity,
                "LeadTimeWeeks": req.lead_time_weeks,
                "RushJob": req.rush_job,
                "Estimator": req.estimator or "Sato-san",
                "TotalPrice_USD": 0.0,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/quote", response_model=QuoteResponse)
def create_quote(req: QuoteRequest) -> QuoteResponse:
    """Generate an AI price estimate for a new part quote."""
    if not _models:
        raise HTTPException(status_code=503, detail="No models loaded")

    df = _request_to_dataframe(req)
    quote_id = f"Q-API-{uuid.uuid4().hex[:8]}"

    # Get predictions from all loaded models
    predictions = {}
    for name, model in _models.items():
        try:
            pred = model.predict(df)
            predictions[name] = float(pred[0])
        except Exception as e:
            logger.warning("Model %s failed: %s", name, e)

    if not predictions:
        raise HTTPException(status_code=500, detail="All models failed")

    # Primary estimate = median across models
    estimate = float(np.median(list(predictions.values())))

    # SHAP explanation (best-effort, tree models only)
    shap_explanation = None
    shap_model_name = None
    for name in ["M2", "M7", "M7c", "M6", "M5"]:
        if name in _models:
            shap_model_name = name
            break

    if shap_model_name:
        try:
            from price_estimator.predict import compute_shap_explanation

            explanation = compute_shap_explanation(_models[shap_model_name], df)
            shap_vals = explanation["shap_values"]
            if shap_vals.ndim > 1:
                shap_vals = shap_vals[0]
            names = explanation["feature_names"]
            indices = np.argsort(np.abs(shap_vals))[::-1][:10]
            shap_explanation = [
                {
                    "feature": names[idx],
                    "contribution": round(float(shap_vals[idx]), 2),
                }
                for idx in indices
            ]
        except Exception as e:
            logger.warning("SHAP computation failed: %s", e)

    # Model range
    model_range = {
        "low": float(min(predictions.values())),
        "high": float(max(predictions.values())),
        "models": predictions,
    }

    # Confidence flags
    confidence_flags = []
    if req.material is None:
        confidence_flags.append("Missing material")
    if req.process is None:
        confidence_flags.append("Missing process")

    if _training_bounds:
        from price_estimator.predict import detect_ood

        ood = detect_ood(df, _training_bounds)
        if ood[0]["is_ood"]:
            confidence_flags.extend(ood[0]["reasons"])

    spread_pct = (model_range["high"] - model_range["low"]) / estimate * 100
    if spread_pct > 20:
        confidence_flags.append(f"High model disagreement: {spread_pct:.1f}% spread")

    # Store in database
    conn = get_db()
    conn.execute(
        "INSERT INTO quotes (quote_id, features, model_price, model_range_low, "
        "model_range_high, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            quote_id,
            json.dumps(req.model_dump()),
            estimate,
            model_range["low"],
            model_range["high"],
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    return QuoteResponse(
        quote_id=quote_id,
        estimate=round(estimate, 2),
        model_range=model_range,
        confidence_flags=confidence_flags,
        shap_explanation=shap_explanation,
    )


@app.post("/quote/{quote_id}/override", response_model=OverrideResponse)
def override_quote(quote_id: str, req: OverrideRequest) -> OverrideResponse:
    """Override an AI quote with a human-determined price."""
    conn = get_db()
    row = conn.execute("SELECT model_price FROM quotes WHERE quote_id = ?", (quote_id,)).fetchone()

    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Quote {quote_id} not found")

    model_price = row[0]
    delta = req.human_price - model_price
    delta_pct = delta / model_price * 100

    override_id = f"OVR-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO overrides (override_id, quote_id, human_price, reason_category, "
        "reason_text, estimator_id, delta_from_model, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            override_id,
            quote_id,
            req.human_price,
            req.reason_category.value if req.reason_category else None,
            req.reason_text,
            req.estimator_id,
            delta,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    return OverrideResponse(
        stored=True,
        override_id=override_id,
        delta_from_model=round(delta, 2),
        delta_pct=round(delta_pct, 2),
    )


@app.get("/quote/{quote_id}", response_model=QuoteDetail)
def get_quote(quote_id: str) -> QuoteDetail:
    """Retrieve a quote with its original estimate and any override."""
    conn = get_db()
    quote_row = conn.execute(
        "SELECT model_price FROM quotes WHERE quote_id = ?", (quote_id,)
    ).fetchone()

    if quote_row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Quote {quote_id} not found")

    model_price = quote_row[0]

    override_row = conn.execute(
        "SELECT human_price, reason_category, reason_text, estimator_id, "
        "delta_from_model, created_at FROM overrides WHERE quote_id = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (quote_id,),
    ).fetchone()

    human_override = None
    final_price = model_price
    if override_row:
        human_override = {
            "human_price": override_row[0],
            "reason_category": override_row[1],
            "reason_text": override_row[2],
            "estimator_id": override_row[3],
            "delta_from_model": override_row[4],
            "overridden_at": override_row[5],
        }
        final_price = override_row[0]

    conn.close()

    return QuoteDetail(
        quote_id=quote_id,
        original_estimate=round(model_price, 2),
        human_override=human_override,
        final_price=round(final_price, 2),
    )
