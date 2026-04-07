"""FastAPI application for the human-in-the-loop quote workflow.

Endpoints (all under /v1):
    POST /v1/quote         - Get an AI price estimate with bands and SHAP explanation
    POST /v1/quote/{id}/override - Override an AI quote with human price
    GET  /v1/quote/{id}    - Retrieve a quote (original + override if any)
"""

import json
import logging
import re
import sqlite3
import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticModel
from pydantic import Field, field_validator

from price_estimator.data import (
    VALID_ESTIMATORS,
    VALID_MATERIALS,
    VALID_PROCESSES,
    VALID_QUANTITIES,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Price Estimator API",
    description="AI-assisted quoting for precision machined aerospace parts",
    version="0.1.0",
)

router = APIRouter(prefix="/v1")


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------


class ErrorDetail(PydanticModel):
    """Structured error response body."""

    code: str
    message: str
    details: dict | None = None


class ErrorResponse(PydanticModel):
    """Wrapper for all error responses."""

    error: ErrorDetail


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Override FastAPI's default 422 to use our error envelope."""
    fields = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"] if part != "body")
        fields.append({"field": loc, "issue": err["msg"]})
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "UNPROCESSABLE_ENTITY",
                "message": f"{len(fields)} validation error(s) in request.",
                "details": {"fields": fields},
            }
        },
    )


def _error_response(status_code: int, code: str, message: str, details: dict | None = None):
    """Raise an HTTPException with our error envelope."""
    raise HTTPException(
        status_code=status_code,
        detail={"error": {"code": code, "message": message, "details": details}},
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


# Printable ASCII + common punctuation (no control chars, no emoji)
_PRINTABLE_RE = re.compile(r"^[\x20-\x7E]+$")


class QuoteRequest(PydanticModel):
    """Input for a new quote request."""

    part_description: str = Field(min_length=1, max_length=200)
    material: str | None = None
    process: str | None = None
    quantity: int = Field(gt=0)
    rush_job: bool = False
    lead_time_weeks: int = Field(ge=1, le=52)
    estimator: str | None = None

    @field_validator("part_description")
    @classmethod
    def validate_part_description(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("part_description must not be blank")
        if not _PRINTABLE_RE.match(v):
            raise ValueError("part_description must contain only printable ASCII characters")
        return v

    @field_validator("material")
    @classmethod
    def validate_material(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_MATERIALS:
            raise ValueError(
                f"Unknown material: '{v}'. Must be one of: {', '.join(VALID_MATERIALS)}"
            )
        return v

    @field_validator("process")
    @classmethod
    def validate_process(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_PROCESSES:
            raise ValueError(
                f"Unknown process: '{v}'. Must be one of: {', '.join(VALID_PROCESSES)}"
            )
        return v

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: int) -> int:
        if v not in VALID_QUANTITIES:
            raise ValueError(f"Invalid quantity: {v}. Must be one of: {VALID_QUANTITIES}")
        return v

    @field_validator("estimator")
    @classmethod
    def validate_estimator(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_ESTIMATORS:
            raise ValueError(
                f"Unknown estimator: '{v}'. Must be one of: {', '.join(VALID_ESTIMATORS)}"
            )
        return v


class QuoteResponse(PydanticModel):
    """Output from a quote request."""

    quote_id: str
    estimate: float
    aggressive_estimate: float | None = None
    conservative_estimate: float | None = None
    typical_range: dict | None = None
    warnings: list[str] = []
    shap_explanation: list[dict] | None = None


class OverrideRequest(PydanticModel):
    """Input for overriding a quote."""

    human_price: float = Field(gt=0, le=10_000_000)
    reason_category: OverrideReasonCategory | None = None
    reason_text: str | None = Field(default=None, max_length=1000)
    estimator_id: str | None = None

    @field_validator("estimator_id")
    @classmethod
    def validate_estimator_id(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_ESTIMATORS:
            raise ValueError(
                f"Unknown estimator: '{v}'. Must be one of: {', '.join(VALID_ESTIMATORS)}"
            )
        return v


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
_prediction_bands: dict | None = None


def set_models(models: dict, training_bounds=None, prediction_bands=None):
    """Set the loaded models for the API to use.

    Called by the serve script after loading serialized models.

    Args:
        models: Dict mapping model name to trained model instance.
        training_bounds: TrainingBounds instance for OOD detection.
        prediction_bands: Dict mapping model name to empirical band dict.
    """
    global _models, _training_bounds, _prediction_bands
    _models = models
    _training_bounds = training_bounds
    _prediction_bands = prediction_bands


ESTIMATORS = ["Sato-san", "Suzuki-san", "Tanaka-san"]

# Models that don't use the estimator feature
DEBIASED_MODELS = {"M0", "M9"}


def _request_to_dataframe(req: QuoteRequest, estimator: str = "Suzuki-san") -> pd.DataFrame:
    """Convert a QuoteRequest to a single-row DataFrame.

    Args:
        req: The quote request.
        estimator: Estimator name to use. When the request doesn't specify
            an estimator, callers should use this to iterate over all
            estimators for range estimation.
    """
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
                "Estimator": req.estimator or estimator,
                "TotalPrice_USD": 0.0,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/quote", response_model=QuoteResponse)
def create_quote(req: QuoteRequest) -> QuoteResponse:
    """Generate an AI price estimate for a new part quote."""
    if not _models:
        _error_response(503, "SERVICE_UNAVAILABLE", "No models loaded")

    estimator_provided = req.estimator is not None
    quote_id = f"Q-API-{uuid.uuid4().hex[:8]}"

    # Get predictions from all loaded models
    predictions = {}
    estimator_range = {}

    if estimator_provided:
        # Estimator known — straightforward prediction
        df = _request_to_dataframe(req)
        for name, model in _models.items():
            try:
                pred = model.predict(df)
                predictions[name] = float(pred[0])
            except Exception as e:
                logger.warning("Model %s failed: %s", name, e)
    else:
        # No estimator — use debiased models directly, run estimator-aware
        # models with all 3 estimators to get a range
        df = _request_to_dataframe(req)  # placeholder estimator for debiased
        per_est_preds: dict[str, list[float]] = {e: [] for e in ESTIMATORS}

        for name, model in _models.items():
            if name in DEBIASED_MODELS:
                try:
                    pred = model.predict(df)
                    predictions[name] = float(pred[0])
                except Exception as e:
                    logger.warning("Model %s failed: %s", name, e)
            else:
                est_preds = {}
                for est in ESTIMATORS:
                    df_est = _request_to_dataframe(req, estimator=est)
                    try:
                        pred = model.predict(df_est)
                        est_preds[est] = float(pred[0])
                        per_est_preds[est].append(float(pred[0]))
                    except Exception as e:
                        logger.warning("Model %s/%s failed: %s", name, est, e)
                if est_preds:
                    predictions[name] = float(np.median(list(est_preds.values())))

        for est in ESTIMATORS:
            if per_est_preds[est]:
                estimator_range[est] = float(np.median(per_est_preds[est]))

    if not predictions:
        _error_response(500, "PREDICTION_FAILED", "All models failed to produce a prediction")

    # Recommendation from top-tier model consensus
    from price_estimator.predict import (
        apply_prediction_band,
        compute_recommendation,
        compute_shap_explanation,
        detect_ood,
    )

    # Use M2's empirical band (best model) for recommendation shifts
    band = _prediction_bands.get("M2") if _prediction_bands else None
    rec = compute_recommendation(predictions, band=band)

    estimate = rec["estimate"]
    aggressive_estimate = rec["win_bid"]
    conservative_estimate = rec["protect_margin"]

    # Typical range from empirical prediction band
    typical_range = None
    if band:
        low, high = apply_prediction_band(estimate, band)
        typical_range = {
            "low": round(low, 2),
            "high": round(high, 2),
            "coverage": band["coverage"],
        }

    # Warnings
    warnings = []
    if req.material is None:
        warnings.append("Missing material — estimate may be less accurate")
    if req.process is None:
        warnings.append("Missing process — estimate may be less accurate")

    if _training_bounds:
        ood = detect_ood(df, _training_bounds)
        if ood[0]["is_ood"]:
            warnings.extend(ood[0]["reasons"])

    if rec["family_divergence"] and rec["family_divergence"]["divergence_pct"] > 10:
        div = rec["family_divergence"]
        warnings.append(
            f"Linear/tree model divergence: {div['divergence_pct']:.0f}% "
            f"(linear ${div['linear_median']:,.0f} vs tree ${div['tree_median']:,.0f}) "
            f"— review recommended"
        )

    if rec["top_tier_spread_pct"] > 15:
        warnings.append(f"Top-tier model spread: {rec['top_tier_spread_pct']:.0f}%")

    if not estimator_provided:
        warnings.append("No estimator provided — using debiased consensus")

    # SHAP explanation (best-effort, tree models only)
    shap_explanation = None
    shap_candidates = ["M9", "M6", "M5"] if not estimator_provided else ["M6", "M7", "M5"]
    for name in shap_candidates:
        if name in _models:
            try:
                explanation = compute_shap_explanation(_models[name], df)
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
                logger.warning("SHAP computation failed for %s: %s", name, e)
            break

    # Store in database
    conn = get_db()
    conn.execute(
        "INSERT INTO quotes (quote_id, features, model_price, model_range_low, "
        "model_range_high, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            quote_id,
            json.dumps(req.model_dump()),
            estimate,
            typical_range["low"] if typical_range else estimate,
            typical_range["high"] if typical_range else estimate,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    return QuoteResponse(
        quote_id=quote_id,
        estimate=round(estimate, 2),
        aggressive_estimate=round(aggressive_estimate, 2) if aggressive_estimate else None,
        conservative_estimate=(round(conservative_estimate, 2) if conservative_estimate else None),
        typical_range=typical_range,
        warnings=warnings,
        shap_explanation=shap_explanation,
    )


@router.post("/quote/{quote_id}/override", response_model=OverrideResponse)
def override_quote(quote_id: str, req: OverrideRequest) -> OverrideResponse:
    """Override an AI quote with a human-determined price."""
    conn = get_db()
    row = conn.execute("SELECT model_price FROM quotes WHERE quote_id = ?", (quote_id,)).fetchone()

    if row is None:
        conn.close()
        _error_response(404, "QUOTE_NOT_FOUND", f"Quote {quote_id} not found")

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


@router.get("/quote/{quote_id}", response_model=QuoteDetail)
def get_quote(quote_id: str) -> QuoteDetail:
    """Retrieve a quote with its original estimate and any override."""
    conn = get_db()
    quote_row = conn.execute(
        "SELECT model_price FROM quotes WHERE quote_id = ?", (quote_id,)
    ).fetchone()

    if quote_row is None:
        conn.close()
        _error_response(404, "QUOTE_NOT_FOUND", f"Quote {quote_id} not found")

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


# Register router after all routes are defined
app.include_router(router)
